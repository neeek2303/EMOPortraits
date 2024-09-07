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

class LocalEncoderMask(nn.Module):
    def __init__(self, use_amp_autocast,
                 gen_upsampling_type,
                 gen_downsampling_type,
                 gen_num_channels,
                 enc_channel_mult,
                 norm_layer_type,
                 num_gpus,
                 gen_max_channels,
                 enc_block_type,
                 gen_activation_type,
                 gen_input_image_size,
                 gen_latent_texture_size,
                 seg_out_channels=16,
                 num_2d_blocks=3,
                 in_channels = 3,
                 ):
        super(LocalEncoderMask, self).__init__()
        self.autocast = use_amp_autocast
        self.upsample_type = gen_upsampling_type
        self.ratio = gen_input_image_size // gen_latent_texture_size
        self.num_2d_blocks = int(math.log(self.ratio, 2))

        spatial_size = gen_input_image_size
        out_channels = int(gen_num_channels * enc_channel_mult)

        setattr(
            self,
            f'mask_from_mask_{spatial_size}px',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                ))

        if norm_layer_type!='bn':
            norm = norm_layer_type
        else:
            norm = 'bn' if num_gpus < 2 else 'sync_bn'

        for i in range(self.num_2d_blocks):
            in_channels = out_channels
            out_channels = min(out_channels * 2, gen_max_channels)
            setattr(
                self,
                f'mask_enc_{i}_block={spatial_size}px',
                utils.blocks[enc_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_layer_type=norm,
                    activation_type=gen_activation_type,
                    resize_layer_type=gen_downsampling_type))
            spatial_size //= 2

        in_channels = out_channels
        out_channels = seg_out_channels

        finale_layers = []
        if enc_block_type == 'res':
            finale_layers += [
                utils.norm_layers[norm](in_channels),
                utils.activations[gen_activation_type](inplace=True)]

        finale_layers += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)]

        self.finale_layers = nn.Sequential(*finale_layers)

    def forward(self, source_img):
        with amp.autocast(enabled=self.autocast):
            s = source_img.shape[2]

            x = getattr(self, f'mask_from_mask_{s}px')(source_img)

            for i in range(self.num_2d_blocks):
                x = getattr(self, f'mask_enc_{i}_block={s}px')(x)
                s //= 2

            x = self.finale_layers(x)

        return x





    #
    #
    #
    #     self.autocast = use_amp_autocast
    #     self.upsample_type = gen_upsampling_type
    #     self.num_2d_blocks = num_2d_blocks
    #
    #     out_channels = int(gen_num_channels * enc_channel_mult)
    #
    #     layers = [nn.Conv2d(
    #         in_channels=3,
    #         out_channels=out_channels,
    #         kernel_size=7,
    #         padding=3)]
    #
    #     if norm_layer_type!='bn':
    #         norm = norm_layer_type
    #     else:
    #         norm = 'bn' if num_gpus < 2 else 'sync_bn'
    #
    #     for i in range(self.num_2d_blocks):
    #         in_channels = out_channels
    #         out_channels = min(out_channels * 2, gen_max_channels)
    #         layers += [
    #             utils.blocks[enc_block_type](
    #                 in_channels=in_channels,
    #                 out_channels=out_channels,
    #                 stride=2 ,
    #                 norm_layer_type=norm,
    #                 activation_type=gen_activation_type,
    #                 resize_layer_type=gen_downsampling_type)]
    #
    #     in_channels = out_channels
    #     out_channels = seg_out_channels
    #
    #     if enc_block_type == 'res':
    #         layers += [
    #             utils.norm_layers[norm](in_channels),
    #             utils.activations[gen_activation_type](inplace=True)]
    #
    #     layers += [
    #         nn.Conv2d(
    #             in_channels=in_channels,
    #             out_channels=out_channels,
    #             kernel_size=1)]
    #
    #     self.encode_2d = nn.Sequential(*layers)
    #
    # def forward(self, source_img):
    #     with amp.autocast(enabled=self.autocast):
    #         feat_2d = self.encode_2d(source_img)
    #
    #     return feat_2d