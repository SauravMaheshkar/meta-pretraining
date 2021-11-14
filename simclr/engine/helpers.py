from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import wandb

from .utils import get_loss, get_loss_simclr

__all__ = [
    "do_pretrain",
    "inner_loop_finetune",
    "do_ft_head",
    "update_lossdict",
    "eval_student",
]

# ======================== PreTraining Function ======================== #


def do_pretrain(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    xis: torch.Tensor,
    xjs: torch.Tensor,
    device: torch.device,
) -> np.number:
    """Perform PreTraining"""

    # Set the Student Model to Train Mode
    student.train()

    # Move Tensors to Device
    xis = xis.to(device)
    xjs = xjs.to(device)

    # Pass through Tensors if teacher isn't None
    if teacher is not None:
        xis = teacher(xis)
        xjs = teacher(xjs)

    # Get SimCLR Loss
    loss: torch.Tensor = get_loss_simclr(student, criterion, xis, xjs)

    # Zero out the optimizer
    optimizer.zero_grad()

    # Backprop
    loss.backward()

    # Step through the optimizer
    optimizer.step()

    return loss.item()  # type: ignore


# ======================== Inner FineTuning Training Loop ======================== #


def inner_loop_finetune(
    student: nn.Module,
    head: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: Iterable,
    val_dl: Iterable,
    num_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """FineTuning Inner Loop"""

    stud_loss: float = 0.0
    stud_acc: float = 0.0

    # Set the Student into evaluation mode
    student.eval()

    # Set the Model into Training mode if teacher exists
    if teacher is not None:
        teacher.train()

    # Iterate through the Training Dataloader
    for i, (x, y) in enumerate(train_dl):

        # Move Tensors to Device
        x = x.to(device)
        y = y.to(device)

        # Get loss
        y_loss, acc = get_loss(student, head, x, y)

        # Step through the optimizer
        optimizer.step(y_loss)

        # Save the Metrics
        stud_loss += y_loss.item()
        stud_acc += acc

        # Break the loop before overflowing the number of steps
        if i == num_steps - 1:
            break

    # Scale the metrics by the number of steps
    stud_loss /= num_steps
    stud_acc /= num_steps

    avgloss: Any = None
    avgacc: Any = None

    # Iterate through the Validation Dataloader
    for i, (x, y) in enumerate(val_dl):

        # Move Tensors to Device
        x = x.to(device)
        y = y.to(device)

        # Get loss
        y_loss, acc = get_loss(student, head, x, y)

        # Save the Metrics
        if avgloss is None:
            avgloss = y_loss
            avgacc = acc
        else:
            avgloss += y_loss
            avgacc += acc
        break

    # FineTuning Gradient
    ft_grad = torch.autograd.grad(
        avgloss,
        list(student.parameters()) + list(head.parameters(time=0)),  # type: ignore
        allow_unused=True,
    )

    return (stud_loss, stud_acc), (avgloss, avgacc), ft_grad, head  # type: ignore


# ======================== Forward Pass through the FineTuning Head ======================== #


def do_ft_head(
    student: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    dl: Iterable,
    device: torch.device,
) -> Tuple:
    """Go through the FineTuning Head"""

    # Set the Student Model into Evaluation Mode
    student.eval()

    # Iterate through the Dataloader
    for x, y in dl:

        # Move Tensors to Device
        x = x.to(device)
        y = y.to(device)

        # Get the metrics
        loss, acc = get_loss(student, head, x, y)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Backprop through the loss
        loss.backward()

        # Step through and then zero out the optimizer
        optimizer.step()
        optimizer.zero_grad()

        break

    return loss.item(), acc


# ======================== Update Loss Dictionary ======================== #


def update_lossdict(lossdict: Dict, update: Dict, action: str = "append") -> Dict:

    # Iterate through the keys of the Update Dictionary
    for k in update.keys():
        # Appened the metric
        if action == "append":
            if k in lossdict:
                lossdict[k].append(update[k])
            else:
                lossdict[k] = [update[k]]

        # Sum the metric value
        elif action == "sum":
            if k in lossdict:
                lossdict[k] += update[k]
            else:
                lossdict[k] = update[k]

        # Raise Error
        else:
            raise NotImplementedError
    return lossdict


# ======================== Evaluate Student ======================== #


def eval_student(
    student: nn.Module, head: nn.Module, dl: Iterable, device: torch.device, split: str
) -> Dict:

    # Put the Student Model in Evaluation Mode
    student.eval()

    net_loss: float = 0.0
    y_pred: List = []
    y_true: List = []
    l_obj = nn.BCEWithLogitsLoss(reduction="sum")

    # Disable Gradient Calculation
    with torch.no_grad():

        # Iterate through the Dataloader
        for data, target in dl:

            y_true.append(target.detach().cpu().numpy())

            # Move Tensors to Device
            data, target = data.to(device), target.to(device)

            # Get Output from the Student Model
            output: torch.Tensor = head(student.logits(data))  # type: ignore

            # Sum the loss across the batch
            net_loss += l_obj(output, target).item()

            y_pred.append(output.detach().cpu().numpy())

    # Sumup Prediction, truths and scale the loss
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    net_loss /= len(dl.dataset)  # type: ignore

    roc_list: List = []

    # Calculate the ROC AUC
    for i in range(y_true.shape[1]):  # type: ignore
        try:
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:  # type: ignore
                roc_list.append(roc_auc_score(y_true[:, i], y_pred[:, i]))  # type: ignore
            else:
                roc_list.append(np.nan)
        except ValueError:
            roc_list.append(np.nan)

    # Print Metrics
    print("Average loss: {:.4f}, {}-AUC: {:.4f}".format(net_loss, split, np.mean(roc_list)))

    # Sync Metrics to Weights and Biases 🔥
    wandb.log({f"{split} AUC": np.mean(roc_list)})
    return {"epoch_loss": net_loss, "auc": roc_list}
