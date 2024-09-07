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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class LocalEncoder(nn.Module):
    def __init__(self, use_amp_autocast,
                 gen_upsampling_type,
                 gen_downsampling_type,
                 gen_input_image_size,
                 gen_latent_texture_size,
                 gen_latent_texture_depth,
                 gen_latent_texture_channels,
                 warp_norm_grad,
                 gen_num_channels,
                 enc_channel_mult,
                 norm_layer_type,
                 num_gpus,
                 gen_max_channels,
                 enc_block_type,
                 gen_activation_type,
                 in_channels,
                 ):
        super(LocalEncoder, self).__init__()
        self.autocast = use_amp_autocast
        self.upsample_type = gen_upsampling_type
        self.downsample_type = gen_downsampling_type
        self.ratio = gen_input_image_size // gen_latent_texture_size
        self.num_2d_blocks = int(math.log(self.ratio, 2))
        self.init_depth = gen_latent_texture_depth
        spatial_size = gen_input_image_size
        if warp_norm_grad:
            self.grid_sample = GridSample(gen_latent_texture_size)
        else:
            self.grid_sample = lambda inputs, grid: F.grid_sample(inputs.float(), grid.float(), padding_mode='reflection')

        out_channels = int(gen_num_channels * enc_channel_mult)

        setattr(
            self,
            f'from_rgb_{spatial_size}px',
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
                f'enc_{i}_block={spatial_size}px',
                utils.blocks[enc_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_layer_type=norm,
                    activation_type=gen_activation_type,
                    resize_layer_type=gen_downsampling_type))
            spatial_size //= 2

        in_channels = out_channels
        out_channels = gen_latent_texture_channels
        finale_layers = []
        if enc_block_type == 'res':
            finale_layers += [
                utils.norm_layers[norm](in_channels),
                utils.activations[gen_activation_type](inplace=True)]

        finale_layers += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * self.init_depth,
                kernel_size=1)]


        self.finale_layers = nn.Sequential(*finale_layers)

    def forward(self, source_img):
        with amp.autocast(enabled=self.autocast):
            s = source_img.shape[2]

            x = getattr(self, f'from_rgb_{s}px')(source_img)

            for i in range(self.num_2d_blocks):
                x = getattr(self, f'enc_{i}_block={s}px')(x)
                s //= 2

            x = self.finale_layers(x)

        return x





# class LocalEncoder(nn.Module):
#     def __init__(self, use_amp_autocast,
#                  gen_upsampling_type,
#                  gen_downsampling_type,
#                  gen_input_image_size,
#                  gen_latent_texture_size,
#                  gen_latent_texture_depth,
#                  gen_latent_texture_channels,
#                  warp_norm_grad,
#                  gen_num_channels,
#                  enc_channel_mult,
#                  norm_layer_type,
#                  num_gpus,
#                  gen_max_channels,
#                  enc_block_type,
#                  gen_activation_type,
#                  dim=1024, depth=16, kernel_size=7, patch_size=8,
#                  # dim=512, depth=32, kernel_size=7, patch_size=4
#                  ):
#         super(LocalEncoder, self).__init__()
#         self.autocast = use_amp_autocast
#         self.upsample_type = gen_upsampling_type
#         self.downsample_type = gen_downsampling_type
#         self.ratio = gen_input_image_size // gen_latent_texture_size
#         self.num_2d_blocks = int(math.log(self.ratio, 2))
#         self.init_depth = gen_latent_texture_depth
#         spatial_size = gen_input_image_size
#         # if warp_norm_grad:
#         #     self.grid_sample = GridSample(gen_latent_texture_size)
#         # else:
#         #     self.grid_sample = lambda inputs, grid: F.grid_sample(inputs.float(), grid.float(), padding_mode='reflection')
#
#         self.dim = dim
#         self.depth = depth
#         # nn.BatchNorm2d(dim)
#         if norm_layer_type!='bn':
#             norm = norm_layer_type
#         else:
#             norm = 'bn' if num_gpus < 2 else 'sync_bn'
#
#         self.fe = nn.Sequential(nn.Conv2d(3, dim, kernel_size=kernel_size, stride=patch_size, padding=2),
#                                 nn.GELU(),
#                                 nn.BatchNorm2d(dim, momentum=0.02))
#
#
#         for i in range(depth):
#             setattr(self, f'leyer{i}', nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=3),
#                     nn.ReLU(),
#                     nn.BatchNorm2d(dim, momentum=0.02),
#                 )),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)))
#
#
#
#         finale_layers = []
#         finale_layers += [
#             nn.PixelShuffle(2),
#             nn.Conv2d(dim//4, dim//2, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.BatchNorm2d(dim//2, momentum=0.02),
#             nn.Conv2d(
#                 in_channels=self.dim//2,
#                 out_channels=1024,
#                 kernel_size=1)]
#
#         self.finale_layers = nn.Sequential(*finale_layers)
#
#     def forward(self, x):
#         # print(x.shape)
#         x = self.fe(x)
#         # print(f'Shape after FE{x.shape}')
#         for i in range(self.depth):
#             layer = getattr(self, f'leyer{i}')
#             x = layer(x)
#             # print(f'Shape after layer {i}: {x.shape}')
#         x = self.finale_layers(x)
#         # print(f'Finale shape {x.shape}')
#         return x
#
#
#
#














    #     layers = [nn.Conv2d(
    #         in_channels=3,
    #         out_channels=out_channels,
    #         kernel_size=7,
    #         padding=3)]
    #
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
    #                 stride=2,
    #                 norm_layer_type=norm,
    #                 activation_type=gen_activation_type,
    #                 resize_layer_type=gen_downsampling_type)]
    #
    #     in_channels = out_channels
    #     out_channels = gen_latent_texture_channels
    #
    #     if enc_block_type == 'res':
    #         layers += [
    #             utils.norm_layers[norm](in_channels),
    #             utils.activations[gen_activation_type](inplace=True)]
    #
    #     layers += [
    #         nn.Conv2d(
    #             in_channels=in_channels,
    #             out_channels=out_channels * self.init_depth,
    #             kernel_size=1)]
    #
    #     self.encode_2d = nn.Sequential(*layers)
    #
    # def forward(self, source_img):
    #     with amp.autocast(enabled=self.autocast):
    #         feat_2d = self.encode_2d(source_img)
    #
    #     return feat_2d
