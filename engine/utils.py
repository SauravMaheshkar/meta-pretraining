import os
from typing import Callable, Literal, Tuple

import torch
import torch.nn as nn

__all__ = ["model_saver", "get_loss", "get_loss_simclr"]

# ======================== Model Saver Function ======================== #


def model_saver(
    epoch, student, head, teacher, pt_opt, pt_sched, ft_opt, hyp_opt, path
) -> None:
    """Saves all important Model Parameters"""
    torch.save(
        {
            "student_sd": student.state_dict(),
            "teacher_sd": teacher.state_dict() if teacher is not None else None,
            "head_sd": head.state_dict(),
            "pt_opt_state_dict": pt_opt.state_dict(),
            "pt_sched_state_dict": pt_sched.state_dict(),
            "ft_opt_state_dict": ft_opt.state_dict(),
            "hyp_opt_state_dict": hyp_opt.state_dict() if teacher is not None else None,
        },
        path + f"/checkpoint_epoch{epoch}.pt",
    )


# ======================== Loss Function ======================== #


def get_loss(
    student: nn.Module, head: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> Tuple:
    head_op: torch.Tensor = head(student.logits(x))  # type: ignore
    pi_stud: torch.Tensor = student(x)
    l_obj: Callable = nn.BCEWithLogitsLoss()
    clf_loss: torch.Tensor = l_obj(head_op, y)
    y_loss_stud: torch.Tensor = (
        clf_loss + 0 * torch.sum(pi_stud[0]) + 0 * torch.sum(pi_stud[1])
    )
    acc_stud: Literal[0] = 0
    return y_loss_stud, acc_stud


# ======================== SimCLR Loss Function ======================== #


def get_loss_simclr(
    student: nn.Module, criterion: Callable, xis: torch.Tensor, xjs: torch.Tensor
) -> torch.Tensor:

    _, zis = student(xis)
    _, zjs = student(xjs)

    loss = criterion(zis, zjs)
    return loss
