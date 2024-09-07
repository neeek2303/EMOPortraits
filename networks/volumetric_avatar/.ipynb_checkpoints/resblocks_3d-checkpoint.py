import torch
from torch import nn
from torch.nn import functional as F
from . import utils
from ..common import layers
from typing import List, Union


class ResBlocks3d(nn.Module):
    def __init__(self,
                 input_channels: int,
                 conv_layer_type: str,
                 num_blocks: int,
                 norm_layer_type: str,
                 activation_type: str,
                 num_gpus: int,
                 channels: Union[List, None],
                 ) -> None:
        super(ResBlocks3d, self).__init__()
        # expansion_factor = 4 if block_type == 'bottleneck' else 1
        # hidden_channels = input_channels // expansion_factor

        layers_ = []

        if channels is None or len(channels)==0:
            channels = [input_channels]*num_blocks

        assert len(channels) == num_blocks

        if norm_layer_type != 'bn':
            norm_3d = norm_layer_type + '_3d'
        else:
            norm_3d = 'bn_3d' if num_gpus < 2 else 'sync_bn'

        input = input_channels

        for i in range(num_blocks):
            out = channels[i]
            layers_.append(utils.blocks['res'](
                in_channels=input,
                out_channels=out,
                stride=1,
                norm_layer_type=norm_3d,
                activation_type=activation_type,
                conv_layer_type=conv_layer_type))
            input = out

            # layers_.append(layers.blocks[block_type](
            #     in_channels=hidden_channels,
            #     out_channels=hidden_channels,
            #     num_layers=num_layers,
            #     expansion_factor=expansion_factor,
            #     kernel_size=3,
            #     stride=1,
            #     norm_layer_type=norm_layer_type,
            #     activation_type=activation_type,
            #     conv_layer_type='conv_3d'))

        self.net = nn.Sequential(*layers_)

    def forward(self, x):

        return self.net(x)