import torch

__all__ = ["MultiTaskHead"]

# ======================== MultitaskHead ======================== #


class MultiTaskHead(torch.nn.Module):
    """
    Creates a MultiTaskHead Pytorch Module

    Attributes
    ----------
    feats : int
        Integer representing the input size (used for in_features in the Linear layer)
    num_classes : int
        Number of classes (used for out_features in the Linear layer)
    """

    def __init__(self, feats: int, num_classes: int = 5) -> None:
        super(MultiTaskHead, self).__init__()
        self.fc_pi: torch.nn.Module = torch.nn.Linear(feats, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_pi: torch.Tensor = self.fc_pi(x)
        return out_pi
