import torch
from torch import nn
from torch.nn import functional as F

from typing import Union, Tuple



class GradScaler(nn.Module):
    def __init__(self, size: Union[int, Tuple[int]]):
        super(GradScaler, self).__init__()

    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        if isinstance(self.size, tuple):
            for i in range(len(self.size)):
                grad_output[..., i] /= self.size[i]

        elif isinstance(self.size, int):
            grad_output /= self.size

        return grad_output


class GridSample(nn.Module):
    def __init__(self, size: Union[int, Tuple[int]]):
        super(GridSample, self).__init__()
        self.scaler = GradScaler(size)

    def forward(self, input, grid, padding_mode='reflection', align_corners=False):
        return F.grid_sample(input, self.scaler(grid), padding_mode=padding_mode, align_corners=align_corners)


def make_grid(h, w, device=torch.device('cpu'), dtype=torch.float32):
    grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    v, u = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([u, v], dim=2).view(1, h, w, 2)

    return grid