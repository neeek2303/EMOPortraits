import torch
from torch import nn
from torch import optim
from torch.cuda import amp
import torch.nn.functional as F
import math
import numpy as np
import itertools
import copy
from torch.cuda import amp
from scipy import linalg
from . import utils
from .utils import ProjectorConv, ProjectorNorm, ProjectorNormLinear, assign_adaptive_conv_params, \
    assign_adaptive_norm_params, Upsample_sg2
import utils.args as args_utils


class Decoder_stage2(nn.Module):
    def __init__(self,
                 eps,
                 image_size,
                 use_amp_autocast,
                 gen_embed_size,
                 gen_adaptive_kernel,
                 gen_adaptive_conv_type,
                 gen_latent_texture_size,
                 in_channels,
                 gen_num_channels,
                 dec_max_channels,
                 gen_use_adanorm,
                 gen_activation_type,
                 gen_use_adaconv,
                 dec_channel_mult,
                 dec_num_blocks,
                 dec_up_block_type,
                 dec_pred_seg,
                 dec_seg_channel_mult,
                 dec_pred_conf,
                 dec_conf_ms_names,
                 dec_conf_names,
                 dec_conf_ms_scales,
                 dec_conf_channel_mult,
                 gen_downsampling_type,
                 num_gpus,
                 norm_layer_type,
                ):
        super(Decoder_stage2, self).__init__()
        self.autocast = use_amp_autocast
        self.adaptive_conv_type = gen_adaptive_conv_type
        num_blocks = dec_num_blocks
        num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
        self.in_channels = in_channels
        out_channels = min(int(gen_num_channels * dec_channel_mult * 2**num_up_blocks), dec_max_channels)
        print(num_up_blocks, out_channels)
        self.gen_max_channels = dec_max_channels
        self.num_gpus = num_gpus
        self.norm_layer_type = norm_layer_type

        if norm_layer_type == 'bn':
            if self.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type
        if gen_use_adanorm:
            norm_layer_type = 'ada_' + norm_layer_type

        # print(norm_layer_type)
        layers = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False)]

        for i in range(num_blocks):
            layers += [
                utils.blocks['res'](
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv')]

        self.res_decoder = nn.Sequential(*layers)
        self.img_decoder = ImageDecoder_stage2(
            use_amp_autocast,
            image_size,
            gen_latent_texture_size,
            gen_use_adanorm,
            gen_num_channels,
            dec_up_block_type,
            gen_activation_type,
            gen_use_adaconv,
            dec_pred_seg,
            dec_seg_channel_mult,
            out_channels,
            self.num_gpus,
            norm_layer_type=norm_layer_type)

        # if dec_pred_conf:
        #
        #     self.conf_ms_names = args_utils.parse_str_to_list(dec_conf_ms_names, sep=',')
        #     self.conf_names = args_utils.parse_str_to_list(dec_conf_names, sep=',')
        #     self.conf_decoder = ConfDecoder(
        #                         use_amp_autocast,
        #                         gen_latent_texture_size,
        #                         dec_conf_ms_scales,
        #                         image_size,
        #                         gen_num_channels,
        #                         dec_conf_channel_mult,
        #                         gen_max_channels,
        #                         gen_activation_type,
        #                         gen_downsampling_type,
        #                         out_channels,
        #                         len(self.conf_ms_names),
        #                         len(self.conf_names),
        #                         num_gpus=self.num_gpus, norm_layer_type=norm_layer_type)
        self.gen_use_adanorm = gen_use_adanorm
        if gen_use_adanorm:
            self.projector = ProjectorNormLinear(net_or_nets=[self.res_decoder, self.img_decoder], eps=eps,
                                           gen_embed_size=32,
                                           gen_max_channels=64)
        else:
            self.projector = ProjectorNorm(net_or_nets=[self.res_decoder, self.img_decoder], eps=eps,
                                           gen_embed_size=gen_embed_size,
                                           gen_max_channels=self.gen_max_channels)

        print(sum(p.numel() for p in self.res_decoder.parameters() if p.requires_grad), sum(p.numel() for p in self.img_decoder.parameters() if p.requires_grad))
        if gen_use_adaconv:
            self.projector_conv = ProjectorConv(net_or_nets=[self.res_decoder, self.img_decoder], eps=eps,
                                                gen_adaptive_kernel=gen_adaptive_kernel,
                                                gen_max_channels=self.gen_max_channels)

    def forward(self, data_dict, embed_dict, feat_2d, input_flip_feat=False, annealing_alpha=0.0, embed=None, stage_two=False, pred_feat=None):
        # with amp.autocast(enabled=self.autocast):
        if self.gen_use_adanorm:
            b, c, es, _ = data_dict['embed'].shape
            params_norm = self.projector(data_dict['embed'].view(b, c, es ** 2))

        else:
            params_norm = self.projector(embed_dict)

        if input_flip_feat:
            # Repeat params for flipped feat
            params_norm_ = []
            for param in params_norm:
                if isinstance(param, tuple):
                    params_norm_.append((torch.cat([p] * 2) for p in param))
                else:
                    params_norm_.append(torch.cat([param] * 2))
        else:
            params_norm_ = params_norm

        assign_adaptive_norm_params([self.res_decoder, self.img_decoder], params_norm_, annealing_alpha)

#         if hasattr(self, 'projector_conv'):
#             params_conv = self.projector_conv(embed_dict)
#
#             if input_flip_feat:
#                 # Repeat params for flipped feat
#                 params_conv_ = []
#                 for param in params_conv:
#                     if isinstance(param, tuple):
#                         params_conv_.append((torch.cat([p] * 2) for p in param))
#                     else:
#                         params_conv_.append(torch.cat([param] * 2))
#             else:
#                 params_conv_ = params_conv
#
#             assign_adaptive_conv_params([self.res_decoder, self.img_decoder], params_conv, self.adaptive_conv_type, annealing_alpha)

        # feat_2d = feat_3d.view(feat_3d.shape[0], self.in_channels, feat_3d.shape[3], feat_3d.shape[4])





        # all_outs = []
        #
        #
        # x = feat_2d
        # feat_2d_ = feat_2d
        # x = list(self.res_decoder.modules())[0](x)
        # all_outs.append(x.clone())
        # for module in self.res_decoder.modules():
        #     if isinstance(module, utils.blocks['res']):
        #         x = module(x)
        #         all_outs.append(x.clone())
        #
        #
        # feat_2d = x

        feat_2d = self.res_decoder(feat_2d)


        img, seg, img_f = self.img_decoder(feat_2d, pred_feat, stage_two=stage_two)

        # # Predict conf
        # if hasattr(self, 'conf_decoder') and self.training and input_flip_feat:
        #     feat, feat_flip = feat_2d.split(feat_2d.shape[0] // 2)
        #
        #     conf_ms, conf_ms_flip, conf, conf_flip = self.conf_decoder(feat, feat_flip)
        #
        #     for conf_ms_k, conf_ms_flip_k, conf_name in zip(conf_ms, conf_ms_flip, self.conf_ms_names):
        #         data_dict[f'{conf_name}_ms'] = conf_ms_k
        #         data_dict[f'{conf_name}_flip_ms'] = conf_ms_flip_k
        #
        #         data_dict[conf_name] = conf_ms_k[0]
        #         data_dict[f'{conf_name}_flip'] = conf_ms_flip_k[0]
        #
        #     for conf_k, conf_flip_k, conf_name in zip(conf, conf_flip, self.conf_names):
        #         data_dict[f'{conf_name}'] = conf_k
        #         data_dict[f'{conf_name}_flip'] = conf_flip_k

        # return img, seg, feat_2d, feat_2d_, all_outs
        if stage_two:
            return img, seg, feat_2d, img_f
        else:
            return img, seg, None, None

# class ImageDecoder(nn.Module):
#     def __init__(self,
#                  use_amp_autocast,
#                  image_size,
#                  gen_latent_texture_size,
#                  gen_use_adanorm,
#                  gen_num_channels,
#                  dec_up_block_type,
#                  gen_activation_type,
#                  gen_use_adaconv,
#                  dec_pred_seg,
#                  dec_seg_channel_mult,
#                  shared_in_channels,
#                  num_gpus,
#                  norm_layer_type):
#         super(ImageDecoder, self).__init__()
#         self.autocast = use_amp_autocast
#         num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
#         out_channels = shared_in_channels
#
#         self.num_gpus =num_gpus
#         if norm_layer_type == 'bn':
#             if self.num_gpus > 1:
#                 norm_layer_type = 'sync_' + norm_layer_type
#
#
#         layers = []
#
#         for i in range(num_up_blocks):
#             in_channels = out_channels
#             out_channels = max(out_channels // 2, gen_num_channels)
#             # out_channels = max(out_channels, gen_num_channels)
#
#             # if out_channels%32!=0:
#             #     c_norm_layer_type = 'gn_24'
#             # else:
#             #     c_norm_layer_type = norm_layer_type
#
#             layers += [
#                 utils.blocks[dec_up_block_type](
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     stride=2,
#                     norm_layer_type=norm_layer_type,
#                     activation_type=gen_activation_type,
#                     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
#                     resize_layer_type='nearest'),
#
#                 # utils.blocks[dec_up_block_type](
#                 #     in_channels=out_channels*2,
#                 #     out_channels=out_channels,
#                 #     norm_layer_type=norm_layer_type,
#                 #     activation_type=gen_activation_type,
#                 #     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),
#             ]
#
#         self.dec_img_blocks = nn.Sequential(*layers)
#
#         layers = [
#             utils.norm_layers[norm_layer_type](out_channels),
#             # utils.norm_layers['bn'](out_channels),
#             utils.activations[gen_activation_type](inplace=True),
#             nn.Conv2d(
#                 in_channels=out_channels,
#                 out_channels=3,
#                 kernel_size=1),
#             nn.Sigmoid()]
#
#         self.dec_img_head = nn.Sequential(*layers)
#
#         if dec_pred_seg:
#             in_channels = shared_in_channels
#             out_channels = int(gen_num_channels * dec_seg_channel_mult * 2**num_up_blocks)
#
#             layers = [
#                 nn.Conv2d(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=1,
#                     bias=False)]
#
#             for i in range(num_up_blocks):
#                 in_channels = out_channels
#                 out_channels = max(out_channels // 2, int(gen_num_channels * dec_seg_channel_mult))
#                 layers += [
#                     utils.blocks[dec_up_block_type](
#                         in_channels=in_channels,
#                         out_channels=out_channels,
#                         stride=2,
#                         norm_layer_type=norm_layer_type,
#                         activation_type=gen_activation_type,
#                         conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
#                         resize_layer_type='nearest')]
#
#             # self.dec_seg_blocks = nn.Sequential(*layers)
#             #
#             # layers = [
#             #     # utils.norm_layers['bn'](out_channels),
#             #     utils.norm_layers[norm_layer_type](out_channels),
#             #     utils.activations[gen_activation_type](inplace=True),
#             #     nn.Conv2d(
#             #         in_channels=out_channels,
#             #         out_channels=1,
#             #         kernel_size=(1,1)),
#             #     nn.Sigmoid()]
#             #
#             # self.dec_seg_head = nn.Sequential(*layers)
#
#     def forward(self, feat, stage_two=False):
#         # with amp.autocast(enabled=self.autocast):
#         img_feat = self.dec_img_blocks(feat)
#         img = self.dec_img_head(img_feat.float())
#
#         seg = None
#         if hasattr(self, 'dec_seg_blocks'):
#             # with amp.autocast(enabled=self.autocast):
#             seg_feat = self.dec_seg_blocks(feat)
#             seg = self.dec_seg_head(seg_feat.float())
#
#         if stage_two:
#             return img, seg, img_feat
#         else:
#             return img, seg, None




class ImageDecoder_stage2(nn.Module):
    def __init__(self,
                 use_amp_autocast,
                 image_size,
                 gen_latent_texture_size,
                 gen_use_adanorm,
                 gen_num_channels,
                 dec_up_block_type,
                 gen_activation_type,
                 gen_use_adaconv,
                 dec_pred_seg,
                 dec_seg_channel_mult,
                 shared_in_channels,
                 num_gpus,
                 norm_layer_type):
        super(ImageDecoder_stage2, self).__init__()
        self.autocast = use_amp_autocast
        num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
        out_channels = shared_in_channels

        self.num_gpus =num_gpus
        if norm_layer_type == 'bn':
            if self.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type


        layers = []

        for i in range(num_up_blocks-1):
            in_channels = out_channels
            out_channels = max(out_channels // 2, gen_num_channels)
            # out_channels = max(out_channels, gen_num_channels)

            # if out_channels%32!=0:
            #     c_norm_layer_type = 'gn_24'
            # else:
            #     c_norm_layer_type = norm_layer_type

            layers += [
                utils.blocks[dec_up_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
                    resize_layer_type='nearest'),

                # utils.blocks[dec_up_block_type](
                #     in_channels=out_channels*2,
                #     out_channels=out_channels,
                #     norm_layer_type=norm_layer_type,
                #     activation_type=gen_activation_type,
                #     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),
            ]


        self.dec_img_blocks = nn.Sequential(*layers)
        in_channels = out_channels
        out_channels = 128
        layers = [utils.blocks[dec_up_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
                    resize_layer_type='nearest')]

        out_channels_feat = [128, 64, 32]
        for i in range(3):
            in_channels = out_channels
            out_channels = out_channels_feat[i]
            # out_channels = max(out_channels, gen_num_channels)

            # if out_channels%32!=0:
            #     c_norm_layer_type = 'gn_24'
            # else:
            #     c_norm_layer_type = norm_layer_type

            layers += [
                utils.blocks[dec_up_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),

                # utils.blocks[dec_up_block_type](
                #     in_channels=out_channels*2,
                #     out_channels=out_channels,
                #     norm_layer_type=norm_layer_type,
                #     activation_type=gen_activation_type,
                #     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),
            ]

        self.dec_img_feat_blocks = nn.Sequential(*layers)

        layers = [
            utils.norm_layers[norm_layer_type](out_channels),
            # utils.norm_layers['bn'](out_channels),
            utils.activations[gen_activation_type](inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=3,
                kernel_size=1),
            # nn.Sigmoid()
            nn.Tanh()
        ]

        self.dec_img_head = nn.Sequential(*layers)


    def forward(self, feat, pred_feat, stage_two=False):
        # with amp.autocast(enabled=self.autocast):
        img_feat = self.dec_img_blocks(feat)
        # print(img_feat.shape, pred_feat.shape)
        # img_feat = self.dec_img_feat_blocks(torch.cat((img_feat, pred_feat), dim=1))
        img_feat = self.dec_img_feat_blocks(img_feat)
        img = self.dec_img_head(img_feat.float())


        if stage_two:
            return img, None, img_feat
        else:
            return img, None, None

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img
#
# class ImageDecoder_SG2(nn.Module):
#     def __init__(self,
#                  use_amp_autocast,
#                  image_size,
#                  gen_latent_texture_size,
#                  gen_use_adanorm,
#                  gen_num_channels,
#                  dec_up_block_type,
#                  gen_activation_type,
#                  gen_use_adaconv,
#                  dec_pred_seg,
#                  dec_seg_channel_mult,
#                  shared_in_channels,
#                  num_gpus,
#                  norm_layer_type):
#         super(ImageDecoder_SG2, self).__init__()
#         self.autocast = use_amp_autocast
#         self.num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
#         out_channels = shared_in_channels
#
#         self.to_rgbs = nn.ModuleList()
#         self.num_gpus =num_gpus
#         if norm_layer_type == 'bn':
#             if self.num_gpus > 1:
#                 norm_layer_type = 'sync_' + norm_layer_type
#
#         self.upsample = Upsample_sg2(kernel=[1, 3, 3, 1])
#         # self.upsample = lambda inputs: F.interpolate(inputs, scale_factor=2, mode='bicubic')
#         self.blocks = nn.ModuleList()
#         self.sigmoid = nn.Sigmoid()
#         layers = [
#             utils.norm_layers[norm_layer_type](out_channels),
#             utils.activations[gen_activation_type](inplace=True),
#             nn.Conv2d(
#                 in_channels=out_channels,
#                 out_channels=3,
#                 kernel_size=1),
#             # nn.Sigmoid()
#             # utils.norm_layers[norm_layer_type](out_channels//4),
#             # utils.activations[gen_activation_type](inplace=True),
#             # nn.Conv2d(
#             #     in_channels=out_channels // 4,
#             #     out_channels=3,
#             #     kernel_size=1),
#
#         ]
#         self.to_rgb1 = nn.Sequential(*layers)
#         # min_list = [192, 96, 48, 24]  #128, 256, 512, 1024
#         # min_list = [256, 128, 64]  # 128, 256, 512, 1024
#         for i in range(self.num_up_blocks):
#             in_channels = out_channels
#             # out_channels = max(out_channels // 2, min_list[i])
#             out_channels = max(out_channels // 2, gen_num_channels)
#             layers = [
#                 utils.blocks[dec_up_block_type](
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     stride=2,
#                     norm_layer_type=norm_layer_type,
#                     activation_type=gen_activation_type,
#                     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
#                     resize_layer_type='blur'),
#
#                 # utils.blocks[dec_up_block_type](
#                 #     in_channels=out_channels,
#                 #     out_channels=out_channels,
#                 #     norm_layer_type=norm_layer_type,
#                 #     activation_type=gen_activation_type,
#                 #     conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),
#             ]
#
#
#             self.blocks.append(nn.Sequential(*layers))
#
#             layers = [
#                 utils.norm_layers[norm_layer_type](out_channels),
#                 # utils.norm_layers['bn'](out_channels),
#                 utils.activations[gen_activation_type](inplace=True),
#                 nn.Conv2d(
#                     in_channels=out_channels,
#                     out_channels=3,
#                     kernel_size=1),
#                 # nn.Tanh()
#             ]
#
#             self.to_rgbs.append(nn.Sequential(*layers))
#
#
#
#     def forward(self, feat):
#         # with amp.autocast(enabled=self.autocast):
#         images = []
#         img = self.to_rgb1(feat.float())
#         # print(img.shape)
#         images.append(img)
#
#         for i in range(self.num_up_blocks):
#             feat = self.blocks[i](feat).float()
#             img = self.to_rgbs[i](feat)
#             images.append(img)
#
#         k=1
#         for i in images[-2::-1]:
#             skip = self.upsample(i)
#             for j in range(k-1):
#                 # print('s', skip.shape)
#                 skip = self.upsample(skip)
#             img = img + skip
#             # print(i.shape, skip.shape, img.shape)
#             k+=1
#
#         img = self.sigmoid(img)
#         # img = norm_ip(img, 0, 1)
#         # img.clamp_(max=1, min=0)
#
#         return img, None
#



class ConfDecoder(nn.Module):
    def __init__(self,
                 use_amp_autocast,
                 gen_latent_texture_size,
                 dec_conf_ms_scales,
                 image_size,
                 gen_num_channels,
                 dec_conf_channel_mult,
                 gen_max_channels,
                 gen_activation_type,
                 gen_downsampling_type,
                 shared_in_channels,
                 num_branches_ms,
                 num_branches,
                 num_gpus,
                 norm_layer_type):
        super(ConfDecoder, self).__init__()
        self.autocast = use_amp_autocast
        self.num_branches_ms = num_branches_ms
        self.num_branches = num_branches
        self.input_spatial_size = gen_latent_texture_size
        self.num_gpus = num_gpus
        num_down_blocks = int(math.log(gen_latent_texture_size * 2**(dec_conf_ms_scales - 1) // image_size, 2))
        num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
        out_channels = int(gen_num_channels * dec_conf_channel_mult * 2**num_up_blocks)

        self.first_layer = nn.Conv2d(shared_in_channels, out_channels, kernel_size=(1, 1), bias=False)

        shared_in_channels = out_channels

        if norm_layer_type == 'bn':
            if self.num_gpus > 1:
                norm_layer_type = 'sync_' + norm_layer_type
        self.conf_down_blocks = nn.ModuleList()

        for i in range(num_down_blocks):
            in_channels = out_channels
            out_channels = min(in_channels * 2, gen_max_channels)

            self.conf_down_blocks += [nn.ModuleList()]

            for k in range(self.num_branches_ms):
                self.conf_down_blocks[-1] += [
                    utils.blocks['conv'](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        norm_layer_type=norm_layer_type,
                        # norm_layer_type='bn',
                        activation_type=gen_activation_type,
                        resize_layer_type=gen_downsampling_type)]

        out_channels = shared_in_channels

        self.conf_up_blocks = nn.ModuleList()

        for i in range(num_up_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(gen_num_channels * dec_conf_channel_mult * 2**i), gen_max_channels)

            self.conf_up_blocks += [nn.ModuleList()]

            for k in range(self.num_branches_ms +  self.num_branches):
                self.conf_up_blocks[-1] += [
                    utils.blocks['conv'](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        norm_layer_type=norm_layer_type,
                        # norm_layer_type='bn',
                        activation_type=gen_activation_type,
                        resize_layer_type='nearest')]

        out_channels = shared_in_channels

        self.conf_same_blocks = nn.ModuleList()

        for k in range(self.num_branches_ms):
            self.conf_same_blocks += [utils.blocks['conv'](
                in_channels=out_channels,
                out_channels=out_channels,
                # norm_layer_type='bn' ,
                norm_layer_type=norm_layer_type,
                activation_type=gen_activation_type)]

        self.conf_head_ms = nn.ModuleDict()
        self.conf_head = nn.ModuleList()
        spatial_size = image_size // 2**dec_conf_ms_scales

        for i in range(dec_conf_ms_scales - 1, -1, -1):
            out_channels = min(int(gen_num_channels * dec_conf_channel_mult * 2**i), gen_max_channels)
            spatial_size *= 2

            self.conf_head_ms['%04d' % spatial_size] = nn.ModuleList()

            for k in range(self.num_branches_ms):
                self.conf_head_ms['%04d' % spatial_size] += [
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=2, # one for "straight" texture, one for flip
                        kernel_size=(1, 1))]

            if spatial_size == image_size:
                for k in range(self.num_branches):
                    self.conf_head += [
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=2,
                            kernel_size=(1, 1))]

    def forward(self, feat, feat_flip):
#         with amp.autocast(enabled=self.autocast):
        assert feat.shape[2] == self.input_spatial_size
        inputs = self.first_layer(torch.cat([feat, feat_flip])) # concat in case batch norm is used
        ms_feat = {}

        # Process feat at the same resolution
        spatial_size = self.input_spatial_size
        ms_feat[spatial_size] = []
        for block in self.conf_same_blocks:
            outputs = block(inputs)
            ms_feat[spatial_size] += [outputs.split(feat.shape[0])]

        # Downsample feat
        outputs = [inputs] * self.num_branches_ms
        for blocks in self.conf_down_blocks:
            spatial_size //= 2
            ms_feat[spatial_size] = []

            for k, (block, output) in enumerate(zip(blocks, outputs)):
                output = block(output)
                outputs[k] = output
                ms_feat[spatial_size] += [output.split(feat.shape[0])]

        # Upsample feat
        spatial_size = self.input_spatial_size
        outputs = [inputs] * self.num_branches_ms + [inputs] * self.num_branches
        for blocks in self.conf_up_blocks:
            spatial_size *= 2
            ms_feat[spatial_size] = []

            for k, (block, output) in enumerate(zip(blocks, outputs)):
                output = block(output)
                outputs[k] = output
                ms_feat[spatial_size] += [output.split(feat.shape[0])]

        # Predict confidences
        conf_ms, conf_ms_flip = [[] for k in range(self.num_branches_ms)], [[] for k in range(self.num_branches_ms)]

        for spatial_size in sorted(self.conf_head_ms.keys())[::-1]: # from highest res to lowest res
            for k, ((feat_, feat_flip_), head) in enumerate(zip(ms_feat[int(spatial_size)][:self.num_branches_ms], self.conf_head_ms[spatial_size])):
                conf_ms[k] += [F.softplus(head(feat_.float())[:, :1]).clamp(0.1, 10)]
                conf_ms_flip[k] += [F.softplus(head(feat_flip_.float())[:, 1:]).clamp(0.1, 10)]

        conf, conf_flip = [], []

        spatial_size = max(ms_feat.keys())
        for head, (feat_, feat_flip_) in zip(self.conf_head, ms_feat[spatial_size][self.num_branches_ms:]):
            conf += [F.softplus(head(feat_.float())[:, :1]).clamp(0.1, 10)]
            conf_flip += [F.softplus(head(feat_flip_.float())[:, 1:]).clamp(0.1, 10)]

        return conf_ms, conf_ms_flip, conf, conf_flip