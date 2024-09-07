import torch
from torch import nn
from torch.nn import functional as F

from ..common import layers



class Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 input_size: int,
                 output_channels: int,
                 min_channels: int,
                 max_channels: int,
                 block_type: str,
                 num_groups: int,
                 num_blocks: int,
                 num_layers: int,
                 norm_layer_type: str,
                 activation_type: str) -> None:
        super(Encoder, self).__init__()
        expansion_factor = 4 if block_type == 'bottleneck' else 1
        spatial_size = input_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        
        setattr(
            self, 
            f'from_rgb_{spatial_size}px', 
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=min_channels * expansion_factor,
                kernel_size=7,
                padding=3,
                bias=False))

        self.num_channels = [min(min_channels * 2**i, max_channels) for i in range(num_groups + 1)]

        for i in range(num_groups):
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

            spatial_size //= 2

        self.downsample = nn.MaxPool2d(2)

        if output_channels != self.num_channels[i + 1] * expansion_factor:
            self.to_feats = nn.Conv2d(
                in_channels=self.num_channels[i + 1] * expansion_factor,
                out_channels=output_channels,
                kernel_size=1,
                bias=False)

    def forward(self, x):
        s = x.shape[2]

        x = getattr(self, f'from_rgb_{s}px')(x)

        for i in range(self.num_groups):
            for j in range(self.num_blocks):
                x = getattr(self, f'group={i}_block={j}_{s}px')(x)
            
            x = self.downsample(x)
            s //= 2

        if hasattr(self, 'to_feats'):
            x = self.to_feats(x)

        return x