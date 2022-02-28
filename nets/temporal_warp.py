from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

__all__ = ["RandWarpAugLearnExMag"]

# ======================== SpatialTransformer ======================== #


class SpatialTransformer(nn.Module):
    """
    Creates a general purpose N-D Spatial Transformer

    Adapted from https://github.com/voxelmorph/voxelmorph

    Attributes
    ----------
    size : Sequence[int]
        Integer representing the size of the Sampling Grid
    mode : str
        Interpolation mode to be used for grid sampling.
        Must be one of 'bilinear', 'nearest' or 'bicubic'
    """

    def __init__(self, size: Sequence[int], mode: str = "bilinear") -> None:
        super(SpatialTransformer, self).__init__()

        assert mode in [
            "bilinear",
            "nearest",
            "bicubic",
        ], "only bilinear, nearest and cubic sampling modes are allowed"

        # Sampling mode
        self.mode = mode

        # Create sampling grid
        vectors: List[torch.Tensor] = [torch.arange(0, s) for s in size]
        grids: Tuple[torch.Tensor, ...] = torch.meshgrid(vectors)
        grid: torch.Tensor = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)  # type: ignore

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:

        # New Locations
        new_locs: torch.Tensor = self.grid + flow
        shape: torch.Size = flow.shape[2:]

        # Normalize grid values to [-1,1] for the Sampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(src.shape) == 3:
            src = src.unsqueeze(-1).repeat(1, 1, 1, 2)
            new_locs = new_locs.unsqueeze(-1).repeat(1, 1, 1, 2)

        # NOTE from the authors 👇👇
        # Move channel dimensions to the last position
        # Not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        samp: torch.Tensor = F.grid_sample(
            src, new_locs, align_corners=True, mode=self.mode
        )
        return samp.squeeze(2)


# ======================== VecInt ======================== #


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.

    Attributes
    ----------
    inshape : Sequence[int]
        Size of the sampling grid used by the Spaital Transformer
    nsteps : int
        Number of steps to scale by
    """

    def __init__(self, inshape: Sequence[int], nsteps: int) -> None:
        super(VecInt, self).__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps

        self.nsteps = nsteps
        self.scale: float = 1.0 / (2**self.nsteps)
        self.transformer: nn.Module = SpatialTransformer(inshape)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


# ======================== ResizeTransformTime ======================== #


class ResizeTransformTime(nn.Module):
    """
    Resizes a Transform, which involves resizing the vector field **and** rescaling it.

    Attributes
    ----------
    inshape : int
        Size of the sampling grid used by the Spaital Transformer
    """

    def __init__(self, sf: float) -> None:
        super(ResizeTransformTime, self).__init__()
        self.sf = sf
        self.mode: str = "linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor: float = self.sf
        if factor < 1:
            x = F.interpolate(
                x,
                align_corners=False,
                scale_factor=factor,
                mode=self.mode,
                recompute_scale_factor=False,  # type: ignore
            )
            x = factor * x
        elif factor > 1:
            x = factor * x
            x = F.interpolate(
                x,
                align_corners=False,
                scale_factor=factor,
                mode=self.mode,
                recompute_scale_factor=False,  # type: ignore
            )
        return x


# ======================== RandWarpAugLearnExMag ======================== #


class RandWarpAugLearnExMag(nn.Module):
    """

    Attributes
    ----------
    inshape : Sequence[int]
        Sequence of integers used for creating Vector fields for warping
    """

    def __init__(
        self,
        inshape: Sequence[int],
        int_steps: int = 5,
        int_downsize: int = 4,
        flow_mag: int = 4,
        smooth_size: int = 25,
    ) -> None:

        super(RandWarpAugLearnExMag, self).__init__()

        self.inshape = inshape

        # Boolean variable that determines whether to resize or not
        resize: bool = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransformTime(1 / int_downsize) if resize else None
        self.fullsize = ResizeTransformTime(int_downsize) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [inshape[0] // int_downsize]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer: nn.Module = SpatialTransformer(inshape)

        # set up smoothing filter
        self.flow_mag: torch.Tensor = torch.nn.parameter.Parameter(
            torch.Tensor([float(flow_mag)])
        )
        self.smooth_size = smooth_size
        self.smooth_pad = smooth_centre = (smooth_size - 1) // 2
        smooth_kernel: np.ndarray = np.zeros(smooth_size)
        smooth_kernel[smooth_centre] = 1
        filt: np.ndarray = gaussian_filter1d(smooth_kernel, smooth_centre).astype(
            np.float32
        )
        self.smooth_kernel: torch.Tensor = torch.from_numpy(filt)

        self.net: nn.Module = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.flow_mag_layer: nn.Module = nn.Linear(32, 1)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        BS, _, _ = source.shape
        source = source.transpose(1, 2)
        x: torch.Tensor = source
        fm: torch.Tensor = 2 * torch.sigmoid(self.flow_mag_layer(self.net(x)))

        fm_std: torch.Tensor = 100 * (self.flow_mag**2)

        flow_field: torch.Tensor = (
            fm.view(BS, 1, 1)
            * fm_std
            * torch.randn(x.shape[0], 1, self.inshape[0]).to(x.device)
        )

        # resize flow for integration
        pos_flow: torch.Tensor = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        # Some Smoothing of the flow field
        pos_flow = F.conv1d(
            pos_flow,
            self.smooth_kernel.view(1, 1, self.smooth_size).to(x.device),
            padding=self.smooth_pad,
            stride=1,
        )

        # warp image with flow field
        y_source: torch.Tensor = self.transformer(source, pos_flow)
        return y_source.transpose(1, 2)
