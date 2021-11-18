from typing import Callable, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from engine.utils import get_loss_simclr

__all__ = [
    "zero_hypergrad",
    "store_hypergrad",
    "neumann_hyperstep_preconditioner",
    "gather_flat_grad",
    "hyper_step",
]

# ======================== Zero Hypergrad ======================== #


def zero_hypergrad(hyper_params: Iterable) -> None:
    """Function to zero out all the hyperparameters"""
    current_index: int = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0
        current_index += p_num_params


# ======================== Zero Hypergrad ======================== #


def store_hypergrad(
    hyper_params: Iterable, total_d_val_loss_d_lambda: torch.Tensor
) -> None:
    current_index: int = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        p.grad = total_d_val_loss_d_lambda[
            current_index : current_index + p_num_params
        ].view(p.shape)
        current_index += p_num_params


# ======================== Gather Flat Gradients ======================== #


def gather_flat_grad(loss_grad: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in loss_grad])


# ======================== Neumann Hyperstep Preconditioner ======================== #


def neumann_hyperstep_preconditioner(
    d_val_loss_d_theta: torch.Tensor,
    d_train_loss_d_w: torch.Tensor,
    elementary_lr: float,
    num_neumann_terms: int,
    model: nn.Module,
    head: nn.Module,
) -> torch.Tensor:

    preconditioner: torch.Tensor = d_val_loss_d_theta.detach()
    counter: torch.Tensor = preconditioner

    i: int = 0
    while i < num_neumann_terms:
        old_counter = counter

        hessian_term = gather_flat_grad(
            grad(
                d_train_loss_d_w,
                list(model.parameters()) + list(head.parameters()),
                grad_outputs=counter.view(-1),
                retain_graph=True,
            )
        )
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner


# ======================== A "Hyper" Step ======================== #


def hyper_step(
    model: nn.Module,
    head: nn.Module,
    teacher: nn.Module,
    hyper_params: Iterable,
    pretrain_loader: Iterable,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    d_val_loss_d_theta: torch.Tensor,
    elementary_lr: float,
    neum_steps: int,
    device: torch.device,
) -> torch.Tensor:

    # Zero out all Hyper Parameters
    zero_hypergrad(hyper_params)

    # Number of weights in our Model and Head
    num_weights: int = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in head.parameters()
    )

    d_train_loss_d_w: torch.Tensor = torch.zeros(num_weights).to(device)

    # Initialize Model and Head
    model.train(), model.zero_grad(), head.train(), head.zero_grad()  # type: ignore

    # NOTE: This should be the pretrain set: gradient of PRETRAINING loss wrt pretrain parameters.
    for _, (xis, xjs) in enumerate(pretrain_loader):

        # Shift Tensors to device
        xis = xis.to(device)
        xjs = xjs.to(device)

        if teacher is not None:
            xis = teacher(xis)
            xjs = teacher(xjs)

        # Calculate Training Loss
        train_loss: torch.Tensor = get_loss_simclr(model, criterion, xis, xjs)
        train_loss = (
            train_loss + train_loss * head(model.logits(xis)).sum() * 0  # type: ignore
        )

        # Zero out the Optimizer
        optimizer.zero_grad()

        d_train_loss_d_w += gather_flat_grad(
            grad(
                train_loss,
                list(model.parameters()) + list(head.parameters()),
                create_graph=True,
                allow_unused=True,
            )
        )
        break
    optimizer.zero_grad()

    # Initialize the preconditioner and counter
    preconditioner: torch.Tensor = d_val_loss_d_theta

    preconditioner = neumann_hyperstep_preconditioner(
        d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, neum_steps, model, head
    )

    indirect_grad: torch.Tensor = gather_flat_grad(
        grad(d_train_loss_d_w, hyper_params, grad_outputs=preconditioner.view(-1))  # type: ignore
    )
    hypergrad = indirect_grad

    zero_hypergrad(hyper_params)
    store_hypergrad(hyper_params, -hypergrad)

    return hypergrad
