from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ecg_simclr_resnet18", "ecg_simclr_resnet34"]

# ======================== Helper Functions ======================== #
# Helper Functions to get 1D Convolutional Layers with,
# varying kernel sizes


def conv5x1(in_channels: int, out_channels: int, stride: int = 1) -> Callable:
    return nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)


def conv9x1(in_channels: int, out_channels: int, stride: int = 1) -> Callable:
    return nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4)


def conv15x1(in_channels: int, out_channels: int, stride: int = 1) -> Callable:
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=15, stride=stride, padding=7
    )


# ======================== Basic Block for Resnet ======================== #


class BasicBlock(nn.Module):
    """
    Creates a "Basic" Block PyTorch Module to be used to create the ResNet architecture

    Attributes
    ----------
    in_channels : int
        Number of input channels for the Block
    out_channels : int
        Number of output channels for the Block
    stride : int
        The value of kernel stride to be used in the 1D Convolution layers with size 15
    downsample : nn.Module
        A Optional PyTorch Sequential / Module to downsample the tensor before the residual connection
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        norm_layer: Callable = nn.BatchNorm1d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1: Callable = conv15x1(in_channels, out_channels, stride)
        self.bn1: Callable = norm_layer(out_channels)
        self.relu: Callable = nn.ReLU(inplace=True)
        self.conv2: Callable = conv15x1(out_channels, out_channels)
        self.bn2: Callable = norm_layer(out_channels)
        self.downsample: Optional[nn.Module] = downsample
        self.stride: int = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Copy to be used in Residual Connection
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling block if provided
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual Connection
        out += identity

        # Non-Lineartiy
        out = self.relu(out)

        return out


# ======================== Resnet ======================== #


class ResNet(nn.Module):
    """
    Creates ResNet PyTorch Module

    Attributes
    ----------
    block : nn.Module
        The "BasicBlock" or something similar which is used as the fundamental Residual
        Block in the Network Architecture
    layers : Sequence[int]
        A Sequence of integers representing the number of layers in the Residual Layers
    zero_init_residual : bool
        Whether to use Zero initialization the last BN in each Residual Branch, shown to
        improve results (https://arxiv.org/abs/1706.02677)
    norm_layer : nn.Module
        The Normalization Layer used in the Network Architecture
    """

    def __init__(
        self,
        block: nn.Module,
        layers: Sequence[int],
        zero_init_residual: bool = False,
        norm_layer: Callable = nn.BatchNorm1d,
    ) -> None:

        super(ResNet, self).__init__()

        self._norm_layer: Callable = norm_layer
        self.zero_init_residual: bool = zero_init_residual
        self.inplanes: int = 32

        self.conv1: nn.Module = nn.Conv1d(
            12, self.inplanes, kernel_size=15, stride=2, padding=3, bias=False
        )
        self.bn1: nn.Module = norm_layer(self.inplanes)
        self.relu: nn.Module = nn.ReLU(inplace=True)

        # Create Residual Layers
        self.layer1: nn.Module = self._make_layer(block, 32, layers[0])
        self.layer2: nn.Module = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3: nn.Module = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4: nn.Module = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool: nn.Module = nn.AdaptiveAvgPool1d(1)

        # Projection Head
        self.proj: nn.Module = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True),
        )

        # Initialize Weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initializes Weights according to the layer type

        * Kaiming Normal Initialization for weights of 1D Convolution Layers
        * Unit Weight and Zero Bias for BatchNorm and GroupNorm Layers
        * If `zero_init_residual` is set to True, Zero initialize the last BN
            in each Residual Branch, shown to improve results (https://arxiv.org/abs/1706.02677)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore

    def _make_layer(
        self, block: nn.Module, planes: int, blocks: int, stride: int = 1
    ) -> nn.Module:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # type: ignore
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(planes),
            )

        layers: List = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # type: ignore
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass through the main body of the net"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the logits by doing a forward pass through the net"""
        x = x.transpose(1, 2)
        return self._forward_impl(x)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        """Compute the output by doing a complete forward pass (including the projection head) through the net"""
        x = x.transpose(1, 2)
        feature: torch.Tensor = self._forward_impl(x)
        out: torch.Tensor = self.proj(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


def _resnet(
    block, layers, pretrained_model: bool, progress: bool, **kwargs
) -> nn.Module:
    """Returns a ResNet architecture as per the parameters"""
    model = ResNet(block, layers, **kwargs)
    return model


def ecg_simclr_resnet18(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> nn.Module:
    """
    Creates a ResNet18 PyTorch Module

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        pretrained_model=pretrained,
        progress=progress,
        **kwargs
    )

def ecg_simclr_resnet34(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> nn.Module:
    """
    Creates a ResNet34 PyTorch Module

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        pretrained_model=pretrained,
        progress=progress,
        **kwargs
    )
