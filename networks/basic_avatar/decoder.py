import torch
from torch import nn
from torch.nn import functional as F

from ..common import layers



class Decoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 output_size: int,
                 min_channels: int,
                 max_channels: int,
                 block_type: str,
                 num_bottleneck_groups: int,
                 num_up_groups: int,
                 num_blocks: int,
                 num_layers: int,
                 norm_layer_type: str,
                 activation_type: str) -> None:
        super(Decoder, self).__init__()
        expansion_factor = 4 if block_type == 'bottleneck' else 1
        spatial_size = output_size // 2**num_up_groups
        self.num_up_groups = num_up_groups
        self.num_blocks = num_blocks

        self.num_channels = [min(min_channels * 2**i, max_channels) for i in reversed(range(num_up_groups + 1))]

        layers_ = []

        if input_channels != self.num_channels[0] * expansion_factor:
            layers_.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=self.num_channels[0] * expansion_factor,
                    kernel_size=1,
                    bias=False))

        for i in range(num_bottleneck_groups):
            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=self.num_channels[0],
                    out_channels=self.num_channels[0],
                    num_layers=num_layers,
                    expansion_factor=expansion_factor,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    activation_type=activation_type))

        self.resblocks_2d = nn.Sequential(*layers_)

        self.upsample = nn.Upsample(scale_factor=2)

        for i in range(num_up_groups):
            spatial_size *= 2

            for j in range(num_blocks):
                setattr(
                    self, 
                    f'group={i}_block={j}_{spatial_size}px', 
                    layers.blocks[block_type](
                        in_channels=self.num_channels[i if j == 0 else i + 1],
                        out_channels=self.num_channels[i + 1],
                        num_layers=num_layers,
                        expansion_factor=expansion_factor,
                        kernel_size=3,
                        stride=1,
                        norm_layer_type=norm_layer_type,
                        activation_type=activation_type))

        setattr(
            self, 
            f'to_rgb_{spatial_size}px', 
            nn.Sequential(
                layers.norm_layers[norm_layer_type](self.num_channels[-1] * expansion_factor, affine=True),
                layers.activations[activation_type](inplace=True),
                nn.Conv2d(
                    in_channels=self.num_channels[-1] * expansion_factor,
                    out_channels=output_channels,
                    kernel_size=1)))

    def forward(self, x):
        s = x.shape[2]

        x = self.resblocks_2d(x)

        for i in range(self.num_up_groups):
            x = self.upsample(x)
            s *= 2
            
            for j in range(self.num_blocks):
                x = getattr(self, f'group={i}_block={j}_{s}px')(x)

        x = getattr(self, f'to_rgb_{s}px')(x)

        return x