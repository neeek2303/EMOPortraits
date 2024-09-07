import torch
from torch import nn
from torch.nn import functional as F
import math

from ..common import layers, params_decomposer
from utils.grid_sample import make_grid



class MotionFieldEstimator(nn.Module):
    def __init__(self,
                 min_channels: int,
                 max_channels: int,
                 input_size: int,
                 output_size: int,
                 block_type: str,
                 num_blocks: list,
                 num_layers: int,
                 norm_layer_type: str,
                 activation_type: str) -> None:
        super(MotionFieldEstimator, self).__init__()
        num_groups = int(math.log(output_size // input_size, 2))
        self.num_channels = [min(min_channels * 2**i, max_channels) for i in reversed(range(num_groups + 1))]
        self.expansion_factor = 4 if block_type == 'bottleneck' else 1

        layers_ = []

        for i in range(1, num_groups + 1):
            layers_.append(nn.Upsample(scale_factor=2))

            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=self.num_channels[i - 1 if j == 0 else i],
                    out_channels=self.num_channels[i],
                    num_layers=num_layers,
                    expansion_factor=self.expansion_factor,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type))

        layers_ += [
            layers.norm_layers[norm_layer_type](self.num_channels[-1] * self.expansion_factor, affine=True),
            layers.activations[activation_type](inplace=True),
            nn.Conv2d(
                in_channels=self.num_channels[-1] * self.expansion_factor,
                out_channels=2,
                kernel_size=1),
            nn.Tanh()]

        self.net = nn.Sequential(*layers_)

        self.register_buffer('identity_grid', make_grid(output_size, output_size))

    def init(self):
        last_conv = list(self.net.modules())[-2]
        
        nn.init.xavier_normal_(last_conv.weight, gain=0.02)
        nn.init.zeros_(last_conv.bias)

    def forward(self, z):
        delta_w = self.net(z).permute(0, 2, 3, 1)

        w = delta_w + self.identity_grid

        return w, delta_w