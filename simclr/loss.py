from typing import Callable

import numpy as np
import torch

__all__ = ["NTXentLoss"]

# ======================== NTXentLoss ======================== #


class NTXentLoss(torch.nn.Module):
    """
    A PyTorch Module for the Normalized Temperature-Scaled Cross-Entropy Loss,
    commonly called referred to as NT-Xent Loss. For a more detailed explanation
    of the loss function please refer to this great blog post by Amit Chaudhary.

    Link -> https://amitness.com/2020/03/illustrated-simclr/#a-calculation-of-cosine-similarity

    Attributes
    ----------
    device : torch.device
        A `torch.device` python object representing the device on which the tensor will be placed
        either cpu or cuda
    batch_size : int
        Batch Size to be used
    temperature : int
        Value for the adjustable temperature parameter, to scale the range of similarity function
    use_cosine_similarity: bool
        Boolean which if true uses the cosine similarity and dot product otherwise
    """

    def __init__(
        self,
        device: torch.device,
        batch_size: int,
        temperature: int,
        use_cosine_similarity: bool,
    ) -> None:
        super(NTXentLoss, self).__init__()
        self.batch_size: int = batch_size
        self.temperature: int = temperature
        self.device: torch.device = device
        self.softmax: Callable = torch.nn.Softmax(dim=-1)
        self.use_cosine_similarity: bool = use_cosine_similarity
        self.mask_samples_from_same_repr: torch.Tensor = (
            self._get_correlated_mask().type(torch.bool)
        )
        self.similarity_function: Callable = self._get_similarity_function()
        self.criterion: Callable = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self) -> Callable:
        if self.use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self) -> torch.Tensor:
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the dot product similarity for the given two Tensors"""
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the Cosine similarity for the given two Tensors"""
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis: torch.Tensor, zjs: torch.Tensor) -> torch.Tensor:
        """Computs the NT-Xent Loss"""
        representations: torch.Tensor = torch.cat([zjs, zis], dim=0)

        similarity_matrix: torch.Tensor = self.similarity_function(
            representations, representations
        )

        # filter out the scores from the positive samples
        l_pos: torch.Tensor = torch.diag(similarity_matrix, self.batch_size)
        r_pos: torch.Tensor = torch.diag(similarity_matrix, -self.batch_size)
        positives: torch.Tensor = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives: torch.Tensor = similarity_matrix[
            self.mask_samples_from_same_repr
        ].view(2 * self.batch_size, -1)

        logits: torch.Tensor = torch.cat((positives, negatives), dim=1)

        # Scale by Temperature 🌡
        logits /= self.temperature

        labels: torch.Tensor = torch.zeros(2 * self.batch_size).to(self.device).long()

        # Calculate the loss between logits using CrossEntropy
        loss: torch.Tensor = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
