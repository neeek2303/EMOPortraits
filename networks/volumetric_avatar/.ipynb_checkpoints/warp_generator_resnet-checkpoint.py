import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.cuda import amp
from . import utils
import itertools
from .utils import ProjectorConv, ProjectorNorm, assign_adaptive_conv_params,assign_adaptive_norm_params
from dataclasses import dataclass

class WarpGenerator(nn.Module):

    @dataclass
    class Config:
        eps: int
        num_gpus: int
        gen_adaptive_conv_type: str
        gen_activation_type: str
        gen_upsampling_type: str
        gen_downsampling_type: str
        gen_dummy_input_size: int
        gen_latent_texture_depth: int
        gen_latent_texture_size: int
        gen_max_channels: int
        gen_num_channels: int
        gen_use_adaconv: bool
        gen_adaptive_kernel: bool
        gen_embed_size: int
        warp_output_size: int
        warp_channel_mult: int
        warp_block_type: str
        norm_layer_type: str
        input_channels: int
        pred_blend_weight: bool = False



    def __init__(self, cfg):
        super(WarpGenerator, self).__init__()

        self.cfg = cfg 
        self.adaptive_conv_type = self.cfg.gen_adaptive_conv_type
        self.gen_activation_type = self.cfg.gen_activation_type
        self.upsample_type = self.cfg.gen_upsampling_type
        self.downsample_type = self.cfg.gen_downsampling_type
        self.input_size = self.cfg.gen_dummy_input_size
        self.output_depth = self.cfg.gen_latent_texture_depth
        self.output_size = self.cfg.gen_latent_texture_size
        # self.pred_blend_weight = pred_blend_weight
        self.warp_output_size = self.cfg.warp_output_size
        self.gen_num_channels = self.cfg.gen_num_channels
        self.warp_channel_mult = self.cfg.warp_channel_mult
        self.gen_max_channels = self.cfg.gen_max_channels
        self.warp_block_type = self.cfg.warp_block_type
        self.norm_layer_type = self.cfg.norm_layer_type


        num_blocks = int(math.log(self.warp_output_size // self.input_size, 2))
        self.num_depth_resize_blocks = int(math.log(self.output_size // self.input_size, 2))

        out_channels = (min(int(self.gen_num_channels * self.warp_channel_mult * 2**num_blocks), self.gen_max_channels))//32*32


        if self.norm_layer_type=='bn':
            if self.cfg.num_gpus > 1:
                norm_layer_type = 'sync_' + self.norm_layer_type

        norm_layer_type = 'ada_' + self.norm_layer_type

        # Initialize net
        self.first_conv = nn.Conv2d(self.cfg.input_channels, out_channels * self.input_size, 1, bias=False)

        self.blocks_3d = nn.ModuleList()

        for i in range(num_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = (min(int(self.gen_num_channels * self.warp_channel_mult * 2**i), self.gen_max_channels))//32*32

            self.blocks_3d += [
                utils.blocks[self.warp_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    norm_layer_type=norm_layer_type,
                    conv_layer_type=('ada_' if self.cfg.gen_use_adaconv else '') + 'conv_3d',
                    activation_type=self.gen_activation_type)]

        if self.warp_block_type == 'res':
            if self.norm_layer_type != 'bn':
                norm_3d = self.norm_layer_type + '_3d'
            else:
                norm_3d = 'bn_3d' if self.cfg.num_gpus < 2 else 'sync_bn'

            self.pre_head = nn.Sequential(
                # utils.norm_layers['bn_3d'](out_channels),
                utils.norm_layers[norm_3d](out_channels),
                utils.activations[self.gen_activation_type](inplace=True))

        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    in_channels=out_channels,
                    out_channels=3,
                    kernel_size=3,
                    padding=1),
                nn.Tanh())])



        self.projector = ProjectorNorm(net_or_nets=self.blocks_3d, eps=self.cfg.eps, gen_embed_size=self.cfg.gen_embed_size,
                                       gen_max_channels=self.gen_max_channels)
        if self.cfg.gen_use_adaconv:
            self.projector_conv = ProjectorConv(net_or_nets=self.blocks_3d, eps=self.cfg.eps,
                                                gen_adaptive_kernel=self.cfg.gen_adaptive_kernel,
                                                gen_max_channels=self.gen_max_channels)

        self.downsample = utils.downsampling_layers[f'{self.downsample_type}_3d'](kernel_size=(2, 1, 1), stride=(2, 1, 1))

        # Greate a meshgrid, which is used for warping calculation from deltas
        grid_s = torch.linspace(-1, 1, self.warp_output_size)
        grid_z = torch.linspace(-1, 1, self.output_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        self.register_buffer('identity_grid', torch.stack([u, v, w], 0)[None])

    def forward(self, embed_dict, annealing_alpha=0.0):# TODO remove annealing_alpha at all

        params_norm = self.projector(embed_dict)
        assign_adaptive_norm_params(self.blocks_3d, params_norm)

        if hasattr(self, 'projector_conv'):
            params_conv = self.projector_conv(embed_dict)
            assign_adaptive_conv_params(self.blocks_3d, params_conv, self.adaptive_conv_type, annealing_alpha)

        b = embed_dict['orig'].shape[0]
        inputs = embed_dict['orig'].view(b, -1, self.input_size, self.input_size)

        size = [self.input_size, self.input_size, self.input_size]
        outputs = self.first_conv(inputs).view(b, -1, *size)

        for i, block in enumerate(self.blocks_3d, 1):
            size[1] *= 2
            size[2] *= 2

            # Calc new depth and if it is upsampled or downsampled
            if i < self.num_depth_resize_blocks:
                depth_new = min(self.output_depth * 2**(self.num_depth_resize_blocks - i), size[1])
            else:
                depth_new = self.output_depth

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

            outputs = block(outputs)

            if depth_resize_type == self.downsample_type:
                outputs = self.downsample(outputs)

        if hasattr(self, 'pre_head'):
            outputs = self.pre_head(outputs.float())

        deltas = self.head[0](outputs)

        warp = (self.identity_grid + deltas).permute(0, 2, 3, 4, 1)

        results = [warp, deltas]

        # if self.pred_blend_weight:
        #     results += [self.head[1](outputs)]

        return results

