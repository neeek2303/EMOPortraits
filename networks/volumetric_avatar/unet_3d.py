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

class Unet3D(nn.Module):

    @dataclass
    class Config:
        eps: float
        num_gpus : int
        gen_embed_size: int
        gen_adaptive_kernel: bool
        gen_use_adanorm: bool
        gen_use_adaconv: bool
        gen_upsampling_type: str
        gen_downsampling_type: str
        gen_dummy_input_size: int
        gen_latent_texture_size: int
        gen_latent_texture_depth: int
        gen_adaptive_conv_type: str
        gen_latent_texture_channels: int
        gen_activation_type: str
        gen_max_channels: int
        warp_norm_grad: bool
        warp_block_type: str
        image_size: int
        norm_layer_type: str
        tex_pred_rgb: bool = False
        tex_use_skip_resblock: bool = True

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.upsample_type = self.cfg.gen_upsampling_type
        self.downsample_type = self.cfg.gen_downsampling_type
        num_3d_blocks = int(math.log(self.cfg.gen_latent_texture_size // self.cfg.gen_dummy_input_size, 2))
        self.init_depth = self.cfg.gen_latent_texture_depth
        self.adaptive_conv_type = self.cfg.gen_adaptive_conv_type
        self.upsample_type = self.cfg.gen_upsampling_type
        self.output_depth = self.cfg.gen_latent_texture_depth
        self.gen_max_channels = self.cfg.gen_max_channels
        self.norm_layer_type = self.cfg.norm_layer_type
        norm_layer_type = self.cfg.norm_layer_type
        
        if self.cfg.warp_norm_grad:
            self.grid_sample = GridSample(self.cfg.gen_latent_texture_size)
        else:
            self.grid_sample = lambda inputs, grid: F.grid_sample(inputs.float(), grid.float(),
                                                                  padding_mode='reflection')

        out_channels = self.cfg.gen_latent_texture_channels

        self.blocks_3d_down = nn.ModuleList()

        if norm_layer_type != 'bn':
            norm_3d = norm_layer_type + '_3d'
        else:
            norm_3d = 'bn_3d' if self.cfg.num_gpus < 2 else 'sync_bn'

        for _ in range(num_3d_blocks):
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.cfg.gen_max_channels)
            self.blocks_3d_down += [
                utils.blocks['res'](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    norm_layer_type=norm_3d,
                    activation_type=self.cfg.gen_activation_type,
                    conv_layer_type='conv_3d')]

        self.downsample = utils.downsampling_layers[self.downsample_type + '_3d'](kernel_size=2, stride=2)
        self.downsample_no_depth = utils.downsampling_layers[self.downsample_type + '_3d'](kernel_size=(1, 2, 2),
                                                                                           stride=(1, 2, 2))

#########################################################################################################################

        num_blocks = int(math.log(self.cfg.gen_latent_texture_size // self.cfg.gen_dummy_input_size, 2))
        self.num_blocks = num_blocks
        out_channels = min(int(self.cfg.gen_latent_texture_channels * 2 ** num_blocks), self.cfg.gen_max_channels)

        self.input_tensor = nn.Parameter(
            torch.empty(1, out_channels, self.cfg.gen_dummy_input_size, self.cfg.gen_dummy_input_size,
                        self.cfg.gen_dummy_input_size))
        nn.init.normal_(self.input_tensor, std=1.0)

        # Initialize net
        self.blocks_3d_up = nn.ModuleList()

        if self.cfg.tex_use_skip_resblock:
            self.skip_blocks_3d_up = nn.ModuleList()

        if norm_layer_type == 'bn':
            if self.cfg.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type

        if self.cfg.gen_use_adanorm:
            norm_layer_type = 'ada_' + norm_layer_type
        elif self.cfg.num_gpus < 2:
            norm_layer_type += '_3d'

        conv_layer_type = 'conv_3d'
        if self.cfg.gen_use_adaconv:
            conv_layer_type = 'ada_' + conv_layer_type


        for i in range(num_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(self.cfg.gen_latent_texture_channels * 2 ** i), self.cfg.gen_max_channels)

            self.blocks_3d_up += [
                utils.blocks['res'](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    norm_layer_type=norm_3d,
                    activation_type=self.cfg.gen_activation_type,
                    conv_layer_type=conv_layer_type)]

            if self.cfg.tex_use_skip_resblock:
                self.skip_blocks_3d_up += [
                    utils.blocks[self.cfg.warp_block_type](
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        # norm_layer_type='bn_3d',
                        norm_layer_type=norm_3d,
                        activation_type=self.cfg.gen_activation_type,
                        conv_layer_type='conv_3d')]

        self.head = nn.Sequential(
            utils.norm_layers[norm_3d](out_channels),
            # utils.norm_layers['bn_3d'](out_channels),
            utils.activations[self.cfg.gen_activation_type](inplace=True),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1))

        if self.cfg.tex_pred_rgb:
            # Auxiliary blocks which predict rgb texture
            num_rgb_blocks = int(math.log(self.cfg.image_size // self.cfg.gen_latent_texture_size, 2))
            self.blocks_rgb = nn.ModuleList()

            for i in range(num_rgb_blocks):
                in_channels = out_channels
                out_channels = out_channels // 2

                self.blocks_rgb += [
                    utils.blocks['res'](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        norm_layer_type=norm_layer_type,
                        activation_type=self.cfg.gen_activation_type,
                        conv_layer_type=conv_layer_type)]

            self.head_rgb = nn.Sequential(
                # utils.norm_layers['bn_3d'](out_channels),
                utils.norm_layers[norm_3d](out_channels),
                utils.activations[self.cfg.gen_activation_type](inplace=True),
                nn.Conv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1))

        net_or_nets = [self.blocks_3d_up]
        if hasattr(self, 'blocks_rgb'):
            net_or_nets += [self.blocks_rgb]

        self.projector = ProjectorNorm(net_or_nets=net_or_nets, eps=self.cfg.eps,
                                       gen_embed_size=self.cfg.gen_embed_size,
                                       gen_max_channels=self.gen_max_channels)
        if self.cfg.gen_use_adaconv:
            self.projector_conv = ProjectorConv(net_or_nets=net_or_nets, eps=self.cfg.eps,
                                                gen_adaptive_kernel=self.cfg.gen_adaptive_kernel,
                                                gen_max_channels=self.gen_max_channels)

        self.downsample_up = utils.downsampling_layers[self.downsample_type + '_3d'](kernel_size=(2, 1, 1),
                                                                                  stride=(2, 1, 1))

    def forward(self, warped_feat_3d, embed_dict=None, align_warp=None, blend_weight=None, annealing_alpha=0.0):

        if blend_weight is not None:
            b, n = blend_weight.shape[:2]
            warped_feat_3d = (warped_feat_3d.view(b, n, *warped_feat_3d.shape[1:]) * blend_weight).sum(1)

        spatial_size = warped_feat_3d.shape[-1]
        outputs = warped_feat_3d
        feat_ms = []
        size = [self.init_depth, spatial_size, spatial_size]

        for i, block in enumerate(self.blocks_3d_down):
            if i < len(self.blocks_3d_down) - 1:
                # Calculate block's output size
                size[1] //= 2
                size[2] //= 2

                depth_new = min(size[0] * 2, size[1])  # depth is increasing at first, but does not exceed height and width

                if depth_new > size[0]:
                    depth_resize_type = self.upsample_type
                elif depth_new < size[0]:
                    depth_resize_type = self.downsample_type
                else:
                    depth_resize_type = 'none'
                size[0] = depth_new

                if depth_resize_type == self.upsample_type:
                    outputs = F.interpolate(outputs, scale_factor=(2, 1, 1), mode=self.upsample_type)

            outputs = block(outputs)
            feat_ms += [outputs]

            if i < len(self.blocks_3d_down) - 1:
                if depth_resize_type == self.downsample_type:
                    outputs = self.downsample(outputs)
                else:
                    outputs = self.downsample_no_depth(outputs)

#############################################################################################
        net_or_nets = [self.blocks_3d_up]
        if hasattr(self, 'blocks_rgb'):
            net_or_nets += [self.blocks_rgb]



        if embed_dict is not None:
            params_norm = self.projector(embed_dict)
            assign_adaptive_norm_params(net_or_nets, params_norm, annealing_alpha)

        if hasattr(self, 'projector_conv'):
            params_conv = self.projector_conv(embed_dict)
            assign_adaptive_conv_params(net_or_nets, params_conv, self.adaptive_conv_type, annealing_alpha)

        assert len(feat_ms) == len(self.blocks_3d_up)

        feat_ms = feat_ms[::-1]  # from low res to high res

        outputs = self.input_tensor.repeat_interleave(feat_ms[0].shape[0], dim=0)

        size = [outputs.shape[2], outputs.shape[3], outputs.shape[4]]

        for i, (block_3d, feat) in enumerate(zip(self.blocks_3d_up, feat_ms), 1):
            size[1] *= 2
            size[2] *= 2

            depth_new = min(self.output_depth * 2 ** (len(self.blocks_3d_up) - i), size[1])
            if depth_new > size[0]:
                depth_resize_type = self.upsample_type
            elif depth_new < size[0]:
                depth_resize_type = self.downsample_type
            else:
                depth_resize_type = 'none'

            size[0] = depth_new

            if depth_resize_type == self.upsample_type:
                outputs = F.interpolate(outputs, scale_factor=2, mode=self.upsample_type)
            else:
                outputs = F.interpolate(outputs, scale_factor=(1, 2, 2), mode=self.upsample_type)

            if hasattr(self, 'skip_blocks_3d_up'):
                outputs_skip = self.skip_blocks_3d_up[i - 1](feat)
            else:
                outputs_skip = feat 

            # print(self.num_blocks, i, depth_new, size, depth_resize_type == self.upsample_type, hasattr(self, 'skip_blocks_3d_up'), outputs.shape, outputs_skip.shape)
            outputs = block_3d(outputs + outputs_skip)

            if depth_resize_type == self.downsample_type:
                outputs = self.downsample_up(outputs)

        latent_texture = self.head(outputs)

        return latent_texture


#
#     def forward(self, warped_feat_3d, embed_dict, align_warp=None, blend_weight=None, annealing_alpha=0.0):
#         if blend_weight is not None:
#             b, n = blend_weight.shape[:2]
#             warped_feat_3d = (warped_feat_3d.view(b, n, *warped_feat_3d.shape[1:]) * blend_weight).sum(1)
#
#         spatial_size = warped_feat_3d.shape[-1]
#         outputs = warped_feat_3d
#
#         for i, block in enumerate(self.blocks_3d_down):
#
#             outputs = block(outputs)
#
# #############################################################################################
#         net_or_nets = [self.blocks_3d_up]
#         if hasattr(self, 'blocks_rgb'):
#             net_or_nets += [self.blocks_rgb]
#
#         params_norm = self.projector(embed_dict)
#         assign_adaptive_norm_params(net_or_nets, params_norm, annealing_alpha)
#
#         if hasattr(self, 'projector_conv'):
#             params_conv = self.projector_conv(embed_dict)
#             assign_adaptive_conv_params(net_or_nets, params_conv, self.adaptive_conv_type, annealing_alpha)
#
#         for i, block_3d in enumerate(self.blocks_3d_up, 1):
#
#             outputs = block_3d(outputs)
#         latent_texture = self.head(outputs)
#
#         return latent_texture
#
#
