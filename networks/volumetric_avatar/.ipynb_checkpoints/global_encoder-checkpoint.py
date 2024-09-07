import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, List

from ..common import layers



class GlobalEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_channels: Union[int, List[int]],
                 output_size: int,
                 min_channels: int,
                 max_channels: int,
                 block_type: str,
                 num_blocks: list,
                 num_layers: int,
                 norm_layer_type: str,
                 activation_type: str) -> None:
        super(GlobalEncoder, self).__init__()
        self.input_size = input_size
        self.output_channels = output_channels
        self.output_size = output_size
        expansion_factor = 4 if block_type == 'bottleneck' else 1
        num_groups = len(num_blocks)
        
        layers_ = [
            nn.Conv2d(
                in_channels=3,
                out_channels=min_channels * expansion_factor,
                kernel_size=7,
                padding=3,
                stride=2,
                bias=False),
            nn.MaxPool2d(2)]

        num_channels = [min_channels] + [min(min_channels * 2**i, max_channels) for i in range(num_groups)]

        for i in range(1, num_groups + 1):
            for j in range(num_blocks[i - 1]):
                layers_.append(layers.blocks[block_type](
                    in_channels=num_channels[i - 1 if j == 0 else i],
                    out_channels=num_channels[i],
                    num_layers=num_layers,
                    expansion_factor=expansion_factor,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type))

            if i < num_groups - 1:
                layers_.append(nn.MaxPool2d(2))

        if isinstance(output_channels, list):
            output_channels = sum(output_channels)

        layers_ += [
            layers.norm_layers[norm_layer_type](num_channels[-1] * expansion_factor),
            layers.activations[activation_type](inplace=True),
            nn.AdaptiveAvgPool2d(self.output_size),
            nn.Conv2d(
                in_channels=num_channels[-1] * expansion_factor,
                out_channels=output_channels,
                kernel_size=1,
                bias=False)]

        self.net = nn.Sequential(*layers_)

    def forward(self, x):
        if x.shape[2] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bicubic')

        y = self.net(x)

        if self.output_size == 1:
            y = y.view(y.shape[0], -1)

        if isinstance(self.output_channels, list):
            return torch.split(y, self.output_channels)
        else:
            return y