# ======================== Imports ======================== #

import argparse
import os
import warnings
from typing import Any, Callable, Dict, List

import higher
import torch
import torch.nn as nn
from numpy import number
from tqdm import tqdm

import wandb
from dataloader import ECGDataSetWrapper
from engine.helpers import (
    do_ft_head,
    do_pretrain,
    eval_student,
    inner_loop_finetune,
    update_lossdict,
)
from engine.utils import model_saver
from hyperparam_utils import gather_flat_grad, hyper_step, zero_hypergrad
from loss import NTXentLoss
from nets.resnet import ecg_simclr_resnet18, ecg_simclr_resnet34
from nets.temporal_warp import RandWarpAugLearnExMag
from nets.wrappers import MultiTaskHead
from utils import set_seed

# Ignore certain warnings for aesthetic output
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="ECG SIMCLR IFT")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pretrain_lr", type=float, default=1e-4)
parser.add_argument("--finetune_lr", type=float, default=1e-4)
parser.add_argument("--hyper_lr", type=float, default=1e-4)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--ex", default=500, type=int)
parser.add_argument("--warmup_epochs", type=int, default=1)
parser.add_argument("--pretrain_steps", type=int, default=10)
parser.add_argument("--finetune_steps", type=int, default=1)
parser.add_argument("--studentarch", type=str, default="resnet18")
parser.add_argument("--teacherarch", type=str, default="warpexmag")
parser.add_argument("--dataset", type=str, default="ecg")
parser.add_argument("--neumann", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--savefol", type=str, default="checkpoints")
parser.add_argument("--save", action="store_false")
parser.add_argument("--no_probs", action="store_true")
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--teach_checkpoint", type=str)

args = parser.parse_args()

# Create a directory to save model checkpoints
os.makedirs(args.savefol, exist_ok=True)

set_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize Normalized Temperature-Scaled Cross-Entropy Loss
nt_xent_criterion = NTXentLoss(
    device, args.batch_size, args.temperature, use_cosine_similarity=True
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args):
    pt_meter = AverageMeter()
    ft_loss_meter = AverageMeter()
    ft_acc_meter = AverageMeter()

    # Create Save Path
    if args.save:
        save_path: str = args.savefol

    if args.teacherarch == "warpexmag":

        # Create the teacher architecture
        teacher: nn.Module = RandWarpAugLearnExMag(inshape=[1024]).to(device)
        # Initialize Hyperparamters, Hyperparamter Optimizer and Hyperparameter Scheduler
        hyp_params: List = list(teacher.parameters())
        hyp_optim: torch.optim.Optimizer = torch.optim.Adam(
            [
                {"params": teacher.net.parameters(), "lr": args.hyper_lr},
                {"params": teacher.flow_mag_layer.parameters(), "lr": args.hyper_lr},
                {"params": [teacher.flow_mag], "lr": 1},
            ]
        )
        hyp_scheduler = None

    else:
        args.teacherarch = None
        teacher = None
        hyp_params = None
        hyp_optim = None
        hyp_scheduler = None

    # Instantiate Datasets and Dataloader
    DSHandle: Callable = ECGDataSetWrapper(args.batch_size)
    pretrain_dl, train_dl, val_dl, test_dl, _, NUM_TASKS_FT = DSHandle.get_data_loaders(
        args
    )

    # Initialize Student and Head
    if args.studentarch == "resnet18":
        student: nn.Module = ecg_simclr_resnet18().to(device)
    elif args.studentarch == "resnet34":
        student: nn.Module = ecg_simclr_resnet34().to(device)
    # Good Error Handling
    else:
        raise NotImplementedError

    head: nn.Module = MultiTaskHead(256, NUM_TASKS_FT).to(device)
    # Initialize Optimizer's and Schedule
    pretrain_optim: torch.optim.Optimizer = torch.optim.Adam(
        student.parameters(), lr=args.pretrain_lr
    )
    pretrain_scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
        pretrain_optim, T_max=args.epochs, eta_min=0, last_epoch=-1
    )
    finetune_optim: torch.optim.Optimizer = torch.optim.Adam(
        head.parameters(), lr=args.finetune_lr
    )

    # Manage Checkpointing
    if args.checkpoint:
        ckpt: Any = torch.load(args.checkpoint)
        student.load_state_dict(ckpt["student_sd"])
        if teacher is not None and ckpt["teacher_sd"] is not None:
            teacher.load_state_dict(ckpt["teacher_sd"])
        head.load_state_dict(ckpt["head_sd"])
        pretrain_optim.load_state_dict(ckpt["pt_opt_state_dict"])
        pretrain_scheduler.load_state_dict(ckpt["pt_sched_state_dict"])
        finetune_optim.load_state_dict(ckpt["ft_opt_state_dict"])
        if teacher is not None and ckpt["hyp_opt_state_dict"] is not None:
            hyp_optim.load_state_dict(ckpt["hyp_opt_state_dict"])
        load_ep = int(os.path.split(args.checkpoint)[-1][16:-3]) + 1
        print(f"Restored from epoch {load_ep}")
    else:
        print("Training from scratch")
        load_ep: int = 0

    if args.teach_checkpoint:
        print("LOADING PT AUG MODEL")
        ckpt: Any = torch.load(args.teach_checkpoint)
        teacher.load_state_dict(ckpt["aug_sd"])
        print("LOAD SUCCESSFUL")

    # Initialize Loss Dictionaries
    stud_pretrain_ld: Dict = {"loss": [], "acc": []}
    stud_finetune_train_ld: Dict = {"loss": [], "acc": []}
    stud_finetune_val_ld: Dict = {"loss": [], "acc": []}
    stud_finetune_test_ld: Dict = {}

    num_finetune_steps: int = args.finetune_steps
    num_neumann_steps: int = args.neumann

    # Start Training
    steps = 0
    print("Starting Training")
    for n in range(load_ep, args.epochs):

        # Create a Progress Bar for better visualization of training
        progress_bar: Any = tqdm(pretrain_dl)

        for _, (xis, xjs) in enumerate(progress_bar):

            # Customize Progress Bar
            progress_bar.set_description("Epoch " + str(n))

            # Zero out hyperparameters of teacher (if needed)
            if teacher is not None:
                zero_hypergrad(hyp_params)

            # Perform pretraining (if needed)
            if n < args.warmup_epochs or teacher is None:

                # Get PreTraining loss
                pt_loss: number = do_pretrain(
                    student,
                    teacher,
                    pretrain_optim,
                    nt_xent_criterion,
                    xis,
                    xjs,
                    device,
                )

                # Sync Metrics to Weights and Biases 🔥
                wandb.log({"PreTraining Loss": pt_loss})

                # Update the PreTraining Meter
                pt_meter.update(pt_loss)

                # Get FineTuning Metrics (if neede)
                if teacher is not None:
                    ft_train_loss, ft_train_acc = do_ft_head(
                        student, head, finetune_optim, train_dl, device
                    )
                else:
                    ft_train_loss, ft_train_acc = 0, 0

                # Sync Metrics to Weights and Biases 🔥
                wandb.log(
                    {
                        "FineTuning Training Loss": ft_train_loss,
                    }
                )

                # Update FineTuning Meter
                ft_loss_meter.update(ft_train_loss)
                ft_acc_meter.update(ft_train_acc)
                ft_val_loss, ft_val_acc, hypg = 0, 0, 0

            else:
                # Get PreTraining loss
                pt_loss = do_pretrain(
                    student,
                    teacher,
                    pretrain_optim,
                    nt_xent_criterion,
                    xis,
                    xjs,
                    device,
                )

                # Sync Metrics to Weights and Biases 🔥
                wandb.log({"PreTraining Loss": pt_loss})

                # Update PreTraining Meter
                pt_meter.update(pt_loss)

                if steps % args.pretrain_steps == 0:
                    with higher.innerloop_ctx(
                        head, finetune_optim, copy_initial_weights=True
                    ) as (fnet, diffopt):
                        (
                            (ft_train_loss, ft_train_acc),
                            (ft_val_loss, ft_val_acc),
                            ft_grad,
                            fnet,
                        ) = inner_loop_finetune(
                            student,
                            fnet,
                            teacher,
                            diffopt,
                            train_dl,
                            val_dl,
                            num_finetune_steps,
                            device,
                        )
                        head.load_state_dict(fnet.state_dict())

                    # Update FineTuning Metrics
                    ft_loss_meter.update(ft_train_loss)
                    ft_acc_meter.update(ft_train_acc)

                    # Sync Metrics to Weights and Biases 🔥
                    wandb.log(
                        {
                            "FineTuning Training Loss": ft_train_loss,
                        }
                    )

                    # Get FineTuning Gradient
                    ft_grad: torch.Tensor = gather_flat_grad(ft_grad)
                    for param_group in pretrain_optim.param_groups:
                        cur_lr = param_group["lr"]
                        break

                    # Take a "Hyper" Step
                    hypg = hyper_step(
                        student,
                        head,
                        teacher,
                        hyp_params,
                        pretrain_dl,
                        nt_xent_criterion,
                        pretrain_optim,
                        ft_grad,
                        cur_lr,
                        num_neumann_steps,
                        device,
                    )
                    hypg = hypg.norm().item()
                    hyp_optim.step()

                else:

                    # Pass through the FineTuning Head
                    ft_train_loss, ft_train_acc = do_ft_head(
                        student, head, finetune_optim, train_dl, device
                    )

                    # Sync Metrics to Weights and Biases 🔥
                    wandb.log(
                        {
                            "FineTuning Training Loss": ft_train_loss,
                        }
                    )

                    # Update FineTuning Metrics
                    ft_loss_meter.update(ft_train_loss)
                    ft_acc_meter.update(ft_train_acc)
                    ft_val_loss, ft_val_acc, hypg = 0, 0, 0

            steps += 1
            progress_bar.set_postfix(
                pretrain_loss="%.4f" % pt_meter.avg,
                finetune_train_loss="%.4f" % ft_loss_meter.avg,
                finetune_train_acc="%.4f" % ft_acc_meter.avg,
            )

            # Update Loss Dictionaries
            stud_pretrain_ld["loss"].append(pt_loss)
            stud_finetune_train_ld["loss"].append(ft_train_loss)
            stud_finetune_train_ld["acc"].append(ft_train_acc)
            stud_finetune_val_ld["loss"].append(ft_val_loss)
            stud_finetune_val_ld["acc"].append(ft_val_acc)

        # Evaluate Student
        if teacher is not None:
            ft_test_ld = eval_student(student, head, test_dl, device, split="Test")
            stud_finetune_test_ld = update_lossdict(stud_finetune_test_ld, ft_test_ld)

            ft_val_ld = eval_student(student, head, val_dl, device, split="Validation")
            stud_finetune_val_ld = update_lossdict(stud_finetune_val_ld, ft_val_ld)

            ft_train_ld = eval_student(student, head, train_dl, device, split="Train")
            stud_finetune_train_ld = update_lossdict(
                stud_finetune_train_ld, ft_train_ld
            )

        if hyp_scheduler is not None:
            hyp_scheduler.step()

        # Reset the Meters
        pt_meter.reset()
        ft_loss_meter.reset()
        ft_acc_meter.reset()

        # Save the Logs
        if args.save:
            tosave = {
                "pretrain_ld": stud_pretrain_ld,
                "finetune_train_ld": stud_finetune_train_ld,
                "finetune_val_ld": stud_finetune_val_ld,
                "finetune_test_ld": stud_finetune_test_ld,
            }
            torch.save(tosave, os.path.join(save_path, "logs.ckpt"))
            if n == args.epochs - 1:
                model_saver(
                    n,
                    student,
                    head,
                    teacher,
                    pretrain_optim,
                    pretrain_scheduler,
                    finetune_optim,
                    hyp_optim,
                    save_path,
                )
                print(f"Saved model at epoch {n}")

                trained_model_artifact = wandb.Artifact(
                    "{}-{}-{}-{}".format(
                        args.seed, args.warmup_epochs, args.epochs, args.ex
                    ),
                    type="{}".format(args.studentarch),
                    description="A {} trained on the PTB-XL ECG dataset using SimCLR using Meta-Parameterized Pre-Training for {} warmup epochs, {} Meta FT examples with random seed {}".format(
                        args.studentarch, args.warmup_epochs, args.ex, args.seed
                    ),
                    metadata=vars(args),
                )

                trained_model_artifact.add_dir(args.savefol)
                wandb.run.log_artifact(trained_model_artifact)
    return (
        student,
        head,
        teacher,
        pretrain_optim,
        pretrain_scheduler,
        finetune_optim,
        hyp_optim,
    )


if __name__ == "__main__":

    wandb.init(
        project="meta-parameterized-pre-training",
        name="{}-{}-{}-{}".format(args.seed, args.warmup_epochs, args.epochs, args.ex),
        entity="sauravmaheshkar",
        job_type="train",
        config=vars(args),
    )
    train(args)
    wandb.run.finish()  # type: ignore
