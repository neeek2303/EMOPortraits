import torch
from torch import nn
from torch.nn import functional as F

from ..common import layers



class VolumeRenderer(nn.Module):
    def __init__(self,
                 input_channels: int,
                 input_depth: int,
                 output_channels: int,
                 mode: str) -> None:
        super(VolumeRenderer, self).__init__()
        self.mode = mode

        if self.mode == 'depth_to_channels':
            self.depth_to_channels = nn.Conv2d(
                in_channels=input_channels * input_depth,
                out_channels=output_channels,
                kernel_size=1,
                bias=False)

        elif self.mode == 'volumetric':
            self.to_sigma = nn.Sequential(
                nn.Conv3d(input_channels, 1, 1, bias=False),
                nn.Softplus())

            self.to_value = nn.Conv3d(input_channels, output_channels, 1, bias=False)

    def forward_depth_to_channels(self, x):
        x_2d = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        y_2d = self.depth_to_channels(x_2d)

        return y_2d

    def forward_volumetric(self, x, depth_values):
        b, _, d, h, w = depth_values.shape

        s = self.to_sigma(x)
        y = self.to_value(x)

        dists = (depth_values[:, :, 1:] - depth_values[:, :, :-1])

        alpha = torch.cat(
            [
                1.0 - torch.exp(-s[:, :, :-1] * dists),
                torch.ones(b, 1, 1, h, w, dtype=x.dtype, device=x.device)
            ],
            dim=2)

        w = alpha * self.cumprod_exclusive(1.0 - alpha + 1e-5, dim=1)

        y_2d = (y * w).sum(2)
        depth_2d = (depth_values * w).sum(2)

        return y_2d, depth_2d

    @staticmethod
    def cumprod_exclusive(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        cumprod = torch.cumprod(tensor, dim)
        cumprod = torch.roll(cumprod, 1, dim)

        first_elements = [slice(0, tensor.shape[i], 1) for i in range(tensor.dim())]
        first_elements[dim] = 0

        cumprod[first_elements] = 1.0

        return cumprod

    def forward(self, x, depth_values=None):
        if self.mode == 'depth_to_channels':
            return self.forward_depth_to_channels(x)

        elif self.mode == 'volumetric':
            return self.forward_volumetric(x, depth_values)