import os
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch.backends import cudnn
from tqdm import tqdm

import wandb
from dataloader import ECGDataSetWrapper
from engine.helpers import eval_student, update_lossdict
from engine.utils import get_loss
from nets.resnet import ecg_simclr_resnet18, ecg_simclr_resnet34
from nets.wrappers import MultiTaskHead
from utils import set_seed

cudnn.deterministic = True
cudnn.benchmark = False

import argparse

parser = argparse.ArgumentParser(description="Eval SIMCLR ECG")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--runseed", type=int, default=0)
parser.add_argument("--ex", type=int, default=500, help="num data points")
parser.add_argument("--finetune_lr", type=float, default=1e-3)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--warmup_epochs", type=int, default=1)
parser.add_argument("--studentarch", type=str, default="resnet34")
parser.add_argument("--training_epochs", type=int, default=50)
parser.add_argument("--dataset", type=str, default="ecg")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--savefol", type=str, default="simclr-ecg-eval")
parser.add_argument("--transfer_eval", action="store_true")
parser.add_argument(
    "--checkpoint", type=str, default="checkpoints/checkpoint_epoch49.pt"
)

args = parser.parse_args()

set_seed(args.seed)

# Create a directory to save model checkpoints
os.makedirs(args.savefol, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.transfer_eval:
    args.savefol += f"/transfereval-{args.ex}ex"
else:
    args.savefol += f"/lineval-{args.ex}ex"


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


def do_train_step(
    student: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple:

    # Set the Student Model in Evaluation Mode
    student.eval()

    # Move Tensors to device
    x = x.to(device)
    y = y.to(device)

    # Get Metrics
    loss, acc = get_loss(student, head, x, y)

    # Zero out the Optimizer
    optimizer.zero_grad()

    # Backprop through the loss
    loss.backward()

    # Step through the optimizer
    optimizer.step()

    return loss.item(), acc


def train(args):

    weights = wandb.run.use_artifact(
        "sauravmaheshkar/meta-parameterized-pre-training/{}-{}-{}-{}:latest".format(
            args.seed, args.warmup_epochs, args.training_epochs, args.ex
        ),
        type="{}".format(args.studentarch),
    )

    model_dir = weights.download(root="weights", recursive=True)

    args.checkpoint = model_dir + "/checkpoint_epoch{}.pt".format(
        (args.training_epochs - 1)
    )

    # Initialize Meters
    ft_loss_meter = AverageMeter()
    ft_acc_meter = AverageMeter()

    # Create Dataset and Dataloaders
    DSHandle: Callable = ECGDataSetWrapper(args.batch_size)
    _, train_dl, val_dl, test_dl, _, NUM_TASKS_FT = DSHandle.get_data_loaders(args)

    # Initialize Student and Head
    if args.studentarch == "resnet18":
        student: nn.Module = ecg_simclr_resnet18().to(device)
    elif args.studentarch == "resnet34":
        student: nn.Module = ecg_simclr_resnet34().to(device)
    # Good Error Handling
    else:
        raise NotImplementedError

    head: nn.Module = MultiTaskHead(256, NUM_TASKS_FT).to(device)

    # Handle checkpointing
    if args.checkpoint is None:
        print("No checkpoint! Training from scratch")
    else:
        ckpt: Any = torch.load(args.checkpoint)
        student.load_state_dict(ckpt["student_sd"], strict=False)
        print("Loading student; not doing pretraining")

    # Initialize Optimizer
    if args.transfer_eval:
        finetune_optim: torch.optim.Optimizer = torch.optim.Adam(
            list(head.parameters()) + list(student.parameters()), lr=args.finetune_lr
        )
    else:
        finetune_optim = torch.optim.Adam(head.parameters(), lr=args.finetune_lr)

    # Initialize Loss Dictionaries
    stud_finetune_train_ld: Dict = {"loss": [], "acc": []}
    stud_finetune_val_ld: Dict = {"loss": [], "acc": []}
    stud_finetune_test_ld: Dict = {}

    # Start Evaluation
    for n in range(args.epochs):

        # Create a Progress Bar for better visualization of training
        progress_bar: Any = tqdm(train_dl)
        for _, (x, y) in enumerate(progress_bar):

            # Customize Progress Bar
            progress_bar.set_description("Finetune Epoch " + str(n))

            # Perform Training Step
            ft_train_loss, ft_train_acc = do_train_step(
                student, head, finetune_optim, x, y
            )

            # Update Meter
            ft_loss_meter.update(ft_train_loss)
            ft_acc_meter.update(ft_train_acc)

            # Sync Metrics to Weights and Biases 🔥
            wandb.log(
                {
                    "FineTuning Training Loss": ft_train_loss,
                }
            )

            # Update the progress bar
            progress_bar.set_postfix(
                finetune_train_loss="%.4f" % ft_loss_meter.avg,
                finetune_train_acc="%.4f" % ft_acc_meter.avg,
            )

            # Update Loss Dictionaries
            stud_finetune_train_ld["loss"].append(ft_train_loss)
            stud_finetune_train_ld["acc"].append(ft_train_acc)

        # Evaluate Student
        ft_test_ld = eval_student(
            student, head, test_dl, device, split="Evaluation-Test"
        )
        stud_finetune_test_ld = update_lossdict(stud_finetune_test_ld, ft_test_ld)

        ft_val_ld = eval_student(
            student, head, val_dl, device, split="Evaluation-Validation"
        )
        stud_finetune_val_ld = update_lossdict(stud_finetune_val_ld, ft_val_ld)

        ft_train_ld = eval_student(
            student, head, train_dl, device, split="Evaluation-Train"
        )
        stud_finetune_train_ld = update_lossdict(stud_finetune_train_ld, ft_train_ld)

        # Save the logs
        tosave = {
            "finetune_train_ld": stud_finetune_train_ld,
            "finetune_val_ld": stud_finetune_val_ld,
            "finetune_test_ld": stud_finetune_test_ld,
        }
        torch.save(tosave, os.path.join(args.savefol, "eval_logs.ckpt"))

        # Reset the Meters
        ft_loss_meter.reset()
        ft_acc_meter.reset()

    evaluation_model_artifact = wandb.Artifact(
        "eval-{}-{}-{}-{}".format(args.seed, args.runseed, args.epochs, args.ex),
        type="Eval-{}".format(args.studentarch),
        description="A finetuned {} evaluated on the PTB-XL ECG dataset which was trained using SimCLR with Meta-Parameterized Pre-Training with {} Meta FT examples and random seed {}".format(
            args.studentarch, args.ex, args.seed
        ),
        metadata=vars(args),
    )

    evaluation_model_artifact.add_dir(args.savefol)
    wandb.run.log_artifact(evaluation_model_artifact)

    return student, head, finetune_optim


if __name__ == "__main__":
    wandb.init(
        project="meta-parameterized-pre-training",
        name="eval-{}-{}-{}-{}".format(args.seed, args.runseed, args.epochs, args.ex),
        entity="sauravmaheshkar",
        job_type="eval",
        config=vars(args),
    )
    train(args)
    wandb.run.finish()  # type: ignore
