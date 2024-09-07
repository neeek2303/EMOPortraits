import torch
from torch import nn

from typing import Union, List
from ..common import layers
from dataclasses import dataclass


class VectorDiscriminator(nn.Module):
    def __init__(self,
                 num_channels: int,
                 max_channels: int,
                 num_blocks: int,
                 input_channels: int,
                 norm_layer='in'):
        super(Discriminator, self).__init__()
        self.num_blocks = num_blocks

        self.in_channels = [min(num_channels * 2 ** (i - 1), max_channels) for i in range(self.num_blocks)]
        self.in_channels[0] = input_channels

        self.out_channels = [min(num_channels * 2 ** i, max_channels) for i in range(self.num_blocks)]
        self.norm_layer = norm_layer
        self.init_networks()

        # print(self.in_channels, self.out_channels)

    def init_networks(self) -> None:
        self.blocks = nn.ModuleList()

        for i in range(self.num_blocks):
            self.blocks.append(
                layers.blocks['conv'](
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels[i],
                    kernel_size=3,
                    padding=1,
                    stride=2 if i < self.num_blocks - 1 else 1,
                    norm_layer_type=self.norm_layer,
                    activation_type='lrelu'))

        self.to_scores = nn.Conv2d(
            in_channels=self.out_channels[-1],
            out_channels=1,
            kernel_size=1)

    def forward(self, inputs):
        outputs = inputs
        features = []

        for block in self.blocks:
            outputs = block(outputs)
            features.append(outputs)

        scores = self.to_scores(outputs)

        return scores, features