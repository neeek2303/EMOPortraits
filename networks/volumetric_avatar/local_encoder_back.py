import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models
from torch.cuda import amp
from argparse import ArgumentParser
import math
from . import GridSample
from . import utils
import numpy as np
import copy
from scipy import linalg
import itertools
from .utils import ProjectorConv, ProjectorNorm, assign_adaptive_conv_params,assign_adaptive_norm_params
from dataclasses import dataclass

class LocalEncoderBack(nn.Module):



    @dataclass
    class Config:
        gen_upsampling_type: str
        gen_downsampling_type: str
        gen_num_channels: int
        enc_channel_mult: int
        norm_layer_type: str
        num_gpus: int
        gen_input_image_size: int
        gen_latent_texture_size: int
        gen_max_channels: int
        enc_block_type: int
        gen_activation_type: str
        seg_out_channels: int
        in_channels= int

    def __init__(self, cfg: Config
                 ):
        super(LocalEncoderBack, self).__init__()

        self.cfg = cfg
        self.upsample_type = self.cfg.gen_upsampling_type
        self.ratio = self.cfg.gen_input_image_size // self.cfg.gen_latent_texture_size
        self.num_2d_blocks = int(math.log(self.ratio, 2))

        spatial_size = self.cfg.gen_input_image_size
        out_channels = int(self.cfg.gen_num_channels * self.cfg.enc_channel_mult)

        setattr(
            self,
            f'seg_from_rgb_{spatial_size}px',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                ))

        if self.cfg.norm_layer_type!='bn':
            norm = self.cfg.norm_layer_type
        else:
            norm = 'bn' if self.cfg.num_gpus < 2 else 'sync_bn'

        for i in range(self.num_2d_blocks):
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.cfg.gen_max_channels)
            setattr(
                self,
                f'seg_enc_{i}_block={spatial_size}px',
                utils.blocks[self.cfg.enc_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_layer_type=norm,
                    activation_type=self.cfg.gen_activation_type,
                    resize_layer_type=self.cfg.gen_downsampling_type))
            spatial_size //= 2

        in_channels = out_channels
        out_channels = self.cfg.seg_out_channels

        finale_layers = []
        if self.cfg.enc_block_type == 'res':
            finale_layers += [
                utils.norm_layers[norm](in_channels),
                utils.activations[self.cfg.gen_activation_type](inplace=True)]

        finale_layers += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)]

        self.finale_layers = nn.Sequential(*finale_layers)

    def forward(self, source_img):

        s = source_img.shape[2]

        x = getattr(self, f'seg_from_rgb_{s}px')(source_img)

        for i in range(self.num_2d_blocks):
            x = getattr(self, f'seg_enc_{i}_block={s}px')(x)
            s //= 2

        x = self.finale_layers(x)

        return x

