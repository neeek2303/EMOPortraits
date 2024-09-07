import copy
import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
import numpy as np
import itertools
from torch.cuda import amp
from networks import basic_avatar, volumetric_avatar
from utils import args as args_utils
from utils import spectral_norm, weight_init, point_transforms
from skimage.measure import label
from .va_losses_and_visuals import calc_train_losses, calc_test_losses, prepare_input_data, MODNET, init_losses
from .va_losses_and_visuals import visualize_data, get_visuals, draw_stickman
from networks.volumetric_avatar.utils import requires_grad, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, \
    _calc_r1_penalty
from scipy import linalg
from datasets.Retinaface import Retinaface


class Model(nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("model")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--norm_layer_type', default='bn', type=str, choices=['bn', 'sync_bn', 'in', 'gn', 'bcn'])
        parser.add_argument('--norm_layer_type_3d', default='bn_3d', type=str)

        parser.add_argument('--estimate_head_pose_from_keypoints', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--head_pose_regressor_path', default='/fsx/nikitadrobyshev/latent-texture-avatar/head_pose_regressor.pth')
        parser.add_argument('--additive_motion', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--use_seg', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_back', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--features_sigm', default=1, type=int)

        parser.add_argument('--use_mix_mask', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_masked_aug', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--resize_depth', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--volume_renderer_mode', default='depth_to_channels', type=str)

        # parser.add_argument('--decoder_num_bottleneck_groups', default=6, type=int)

        parser.add_argument('--init_type', default='kaiming')
        parser.add_argument('--init_gain', default=0.0, type=float)

        parser.add_argument('--dis_num_channels', default=64, type=int)
        parser.add_argument('--dis_max_channels', default=512, type=int)
        # parser.add_argument('--dis_num_blocks', default=4, nargs="+", type=int)
        parser.add_argument('--dis_num_blocks', default=4, type=int)
        parser.add_argument('--dis_num_scales', default=1, type=int)

        parser.add_argument('--use_hq_disc', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dis2_num_channels', default=64, type=int)
        parser.add_argument('--dis2_max_channels', default=512, type=int)
        parser.add_argument('--dis2_num_blocks', default=4, type=int)
        parser.add_argument('--dis2_num_scales', default=2, type=int)

        parser.add_argument('--dis_init_type', default='xavier')
        parser.add_argument('--dis_init_gain', default=0.02, type=float)

        parser.add_argument('--adversarial_weight', default=1.0, type=float)
        parser.add_argument('--mix_gen_adversarial', default=1.0, type=float)
        parser.add_argument('--feature_matching_weight', default=60.0, type=float)
        parser.add_argument('--vgg19_weight', default=20.0, type=float)
        parser.add_argument('--vgg19_face', default=0.0, type=float)
        parser.add_argument('--vgg19_face_mixing', default=0.0, type=float)
        parser.add_argument('--vgg19_fv_mix', default=0.0, type=float)
        parser.add_argument('--resnet18_fv_mix', default=0.0, type=float)
        parser.add_argument('--mix_losses_start', default=4, type=int)
        parser.add_argument('--contr_losses_start', default=1, type=int)

        parser.add_argument('--face_resnet', default=0.0, type=float)

        parser.add_argument('--vgg19_emotions', default=0.0, type=float)
        parser.add_argument('--resnet18_emotions', default=0.0, type=float)
        parser.add_argument('--landmarks', default=0.0, type=float)

        parser.add_argument('--l1_weight', default=0.0, type=float)
        parser.add_argument('--l1_back', default=0.0, type=float)
        parser.add_argument('--cycle_idn', default=0.0, type=float)
        parser.add_argument('--cycle_exp', default=0.0, type=float)
        parser.add_argument('--vgg19_weight_cycle_idn', default=0.0, type=float)
        parser.add_argument('--vgg19_face_cycle_idn', default=0.0, type=float)
        parser.add_argument('--vgg19_weight_cycle_exp', default=0.0, type=float)
        parser.add_argument('--vgg19_face_cycle_exp', default=0.0, type=float)
        parser.add_argument('--stm', default=1.0, type=float)
        parser.add_argument('--pull_idt', default=0.0, type=float)
        parser.add_argument('--pull_exp', default=0.0, type=float)
        parser.add_argument('--push_idt', default=0.0, type=float)
        parser.add_argument('--push_exp', default=0.0, type=float)
        parser.add_argument('--contrastive_exp', default=0.0, type=float)
        parser.add_argument('--contrastive_idt', default=0.0, type=float)

        parser.add_argument('--only_cycle_embed', default='True', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--gaze_weight', default=0.0, type=float)
        parser.add_argument('--vgg19_num_scales', default=4, type=int)
        parser.add_argument('--warping_reg_weight', default=0.0, type=float)

        # parser.add_argument('--spn_networks', default = '')
        parser.add_argument('--spn_networks',
                            default='local_encoder, local_encoder_seg, local_encoder_mask, idt_embedder, expression_embedder, xy_generator, uv_generator, warp_embed_head_orig, pose_embed_decode, pose_embed_code, volume_process_net, volume_source_net, volume_pred_net, decoder, backgroung_adding, background_process_net')
        parser.add_argument('--ws_networks',
                            default='local_encoder, local_encoder_seg, local_encoder_mask, idt_embedder, expression_embedder, xy_generator, uv_generator, warp_embed_head_orig,  pose_embed_decode, pose_embed_code, volume_process_net, volume_source_net, volume_pred_net, decoder, backgroung_adding, background_process_net')
        parser.add_argument('--spn_layers', default='conv2d, conv3d, linear, conv2d_ws, conv3d_ws')
        parser.add_argument('--use_sn', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_ws', default='False', type=args_utils.str2bool, choices=[True, False])

        # Optimization options
        parser.add_argument('--gen_opt_type', default='adam')
        parser.add_argument('--gen_lr', default=2e-4, type=float)
        parser.add_argument('--gen_beta1', default=0.5, type=float)
        parser.add_argument('--gen_beta2', default=0.999, type=float)

        parser.add_argument('--gen_weight_decay', default=1e-4, type=float)
        parser.add_argument('--gen_weight_decay_layers', default='conv2d')
        parser.add_argument('--gen_weight_decay_params', default='weight')
        parser.add_argument('--grid_sample_padding_mode', default='zeros')

        parser.add_argument('--gen_shd_type', default='cosine')
        parser.add_argument('--gen_shd_max_iters', default=2.5e5, type=int)
        parser.add_argument('--gen_shd_lr_min', default=1e-6, type=int)

        parser.add_argument('--dis_opt_type', default='adam')
        parser.add_argument('--dis_lr', default=2e-4, type=float)
        parser.add_argument('--dis_beta1', default=0.5, type=float)
        parser.add_argument('--dis_beta2', default=0.999, type=float)

        parser.add_argument('--dis_shd_type', default='cosine')
        parser.add_argument('--dis_shd_max_iters', default=2.5e5, type=int)
        parser.add_argument('--dis_shd_lr_min', default=4e-6, type=int)
        parser.add_argument('--eps', default=1e-8, type=float)

        # Gen parametres
        parser.add_argument('--gen_num_channels', default=32, type=int)
        parser.add_argument('--gen_max_channels', default=512, type=int)
        parser.add_argument('--dec_max_channels', default=512, type=int)

        parser.add_argument('--gen_max_channels_unet3d', default=512, type=int)
        parser.add_argument('--gen_max_channels_loc_enc', default=512, type=int)

        parser.add_argument('--gen_activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--gen_downsampling_type', default='avgpool', type=str)
        parser.add_argument('--gen_upsampling_type', default='trilinear', type=str)
        parser.add_argument('--gen_pred_flip', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--gen_pred_mixing', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--less_em', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--cat_em', default='False', type=args_utils.str2bool, choices=[True, False])


        # Vol render
        parser.add_argument('--l1_vol_rgb', default=0.0, type=float)
        parser.add_argument('--l1_vol_rgb_mix', default=0.0, type=float)
        parser.add_argument('--start_vol_rgb', default=0, type=int)
        parser.add_argument('--squeeze_dim', default=0, type=int)
        parser.add_argument('--coarse_num_sample', default=48, type=int)
        parser.add_argument('--hidden_vol_dec_dim', default=448, type=int)
        parser.add_argument('--targ_vol_loss_scale', default=0.0, type=float)
        parser.add_argument('--num_layers_vol_dec', default=2, type=int)



        # parser.add_argument('--gen_input_image_size', default=256, type=int)
        parser.add_argument('--idt_image_size', default=256, type=int)
        parser.add_argument('--exp_image_size', default=256, type=int)
        parser.add_argument('--image_additional_size', default=None, type=int)

        parser.add_argument('--gen_latent_texture_size', default=64, type=int)
        parser.add_argument('--gen_latent_texture_depth', default=16, type=int)
        parser.add_argument('--gen_latent_texture_channels', default=64, type=int)

        parser.add_argument('--latent_volume_channels', default=64, type=int)
        parser.add_argument('--latent_volume_size', default=64, type=int)
        parser.add_argument('--latent_volume_depth', default=16, type=int)
        parser.add_argument('--source_volume_num_blocks', default=0, type=int)
        parser.add_argument('--pred_volume_num_blocks', default=0, type=int)
        parser.add_argument('--use_tensor', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--old_mix_pose', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--green', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--random_theta', default='False', type=args_utils.str2bool, choices=[True, False])


        parser.add_argument('--reduce_em_ratio', default=2, type=int)
        parser.add_argument('--gen_embed_size', default=4, type=int)
        parser.add_argument('--gen_dummy_input_size', default=4, type=int)

        parser.add_argument('--gen_use_adanorm', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--gen_use_adaconv', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--gen_adaptive_conv_type', default='sum', type=str, choices=['sum', 'mul'])
        parser.add_argument('--gen_adaptive_kernel', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--sep_test_losses', default='True', type=args_utils.str2bool, choices=[True, False])

        # Options for transition between non-adaptive and adaptive params
        parser.add_argument('--gen_adaptive_use_annealing', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--gen_adaptive_annealing_type', default='cos', type=str, choices=['lin', 'cos'])
        parser.add_argument('--gen_adaptive_annealing_max_iter', default=1e5, type=int)

        # XY-gen and UV-gen options
        parser.add_argument('--warp_block_type', default='res', type=str, choices=['res', 'conv'])
        parser.add_argument('--warp_channel_mult', default=1.0, type=float)
        parser.add_argument('--warp_norm_grad', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--warp_output_size', default=64, type=int)

        # Encoder options
        parser.add_argument('--enc_init_num_channels', default=512, type=int)
        parser.add_argument('--enc_channel_mult', default=2.0, type=float)
        parser.add_argument('--enc_block_type', default='res', type=str, choices=['res', 'conv'])

        # Texture gen options
        parser.add_argument('--tex_use_skip_resblock', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--tex_pred_rgb', default='False', type=args_utils.str2bool, choices=[True, False])

        # Decoder options
        parser.add_argument('--dec_num_blocks', default=8, type=int)
        parser.add_argument('--dec_channel_mult', default=2.0, type=float)
        parser.add_argument('--dec_up_block_type', default='res', type=str, choices=['res', 'conv'])
        parser.add_argument('--gen_max_channels', default=512, type=int)
        parser.add_argument('--dec_pred_seg', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dec_seg_channel_mult', default=1.0, type=float)

        parser.add_argument('--dec_pred_conf', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dec_bigger', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--sep_train_losses', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dec_conf_ms_names', default='target_vgg19_conf, target_fem_conf', type=str)
        parser.add_argument('--dec_conf_names', default='', type=str)
        parser.add_argument('--dec_conf_ms_scales', default=5, type=int)
        parser.add_argument('--dec_conf_channel_mult', default=1.0, type=float)
        parser.add_argument('--volume_rendering', default='False', type=args_utils.str2bool, choices=[True, False])

        # Identical encoding
        parser.add_argument('--separate_idt', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--idt_backbone', default='resnet50', type=str)
        parser.add_argument('--idt_lr_mult', default=1.0, type=float)
        parser.add_argument('--idt_output_channels', default=512, type=int)
        parser.add_argument('--idt_output_size', default=4, type=int)

        # Expression encoding
        parser.add_argument('--lpe_head_backbone', default='resnet18', type=str)
        parser.add_argument('--lpe_face_backbone', default='resnet18', type=str)
        parser.add_argument('--lpe_lr_mult', default=1.0, type=float)
        parser.add_argument('--lpe_final_pooling_type', default='avg', type=str, choices=['avg', 'transformer'])
        parser.add_argument('--lpe_output_channels', default=512, type=int)
        parser.add_argument('--lpe_output_size', default=4, type=int)
        parser.add_argument('--lpe_head_transform_sep_scales', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--num_b_negs', default=1, type=int)

        parser.add_argument('--use_stylegan_d', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dis_stylegan_lr', default=2e-4, type=float)
        parser.add_argument('--barlow', default=0, type=float)

        parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
        parser.add_argument('--stylegan_weight', default=1.0, type=float)
        parser.add_argument("--r1", type=float, default=0.0, help="weight of the r1 regularization")

        return parser_out

    def __init__(self, args, training=True):
        super(Model, self).__init__()
        self.args = args
        self.num_source_frames = args.num_source_frames
        self.num_target_frames = args.num_target_frames
        self.resize_d = lambda img: F.interpolate(img, mode='bilinear',
                                           size=(224, 224),
                                           align_corners=False)

        self.resize_u = lambda img: F.interpolate(img, mode='bilinear',
                                           size=(256, 256),
                                           align_corners=False)
        self.embed_size = args.gen_embed_size
        self.num_source_frames = args.num_source_frames  # number of identities per batch
        self.embed_size = args.gen_embed_size
        self.pred_seg = args.dec_pred_seg
        self.use_stylegan_d = args.use_stylegan_d
        self.bn = nn.BatchNorm1d(512, affine=False)
        self.thetas_pool = []
        if self.pred_seg:
            self.seg_loss = nn.BCELoss()
        self.pred_flip = args.gen_pred_flip
        self.pred_mixing = args.gen_pred_mixing
        assert self.num_source_frames == 1, 'No support for multiple sources'
        self.background_net_input_channels = 64

        # self.pred_mixing = args.gen_pred_mixing
        self.weights = {
            'barlow': args.barlow,
            'adversarial': args.adversarial_weight,
            'mix_gen_adversarial':args.mix_gen_adversarial,
            'feature_matching': args.feature_matching_weight,
            'vgg19': args.vgg19_weight,
            'warping_reg': args.warping_reg_weight,
            'gaze': args.gaze_weight,
            'l1_weight': args.l1_weight,
            'l1_back': args.l1_back,
            'vgg19_face': args.vgg19_face,
            'vgg19_face_mixing': args.vgg19_face_mixing,
            'resnet18_fv_mix': args.resnet18_fv_mix,
            'vgg19_fv_mix': args.vgg19_fv_mix,

            'face_resnet': args.face_resnet,
            'vgg19_emotions': args.vgg19_emotions,
            'landmarks': args.landmarks,
            'resnet18_emotions': args.resnet18_emotions,
            'cycle_idn': args.cycle_idn,
            'cycle_exp': args.cycle_exp,
            'l1_vol_rgb':args.l1_vol_rgb,
            'l1_vol_rgb_mix':args.l1_vol_rgb_mix,
            'vgg19_face_cycle_idn': args.vgg19_face_cycle_idn,
            'vgg19_cycle_idn': args.vgg19_weight_cycle_idn,

            'vgg19_face_cycle_exp': args.vgg19_face_cycle_exp,
            'vgg19_cycle_exp': args.vgg19_weight_cycle_exp,

            'stm':args.stm,
            'pull_idt': args.pull_idt,
            'pull_exp': args.pull_exp,
            'push_idt': args.push_idt,
            'push_exp': args.push_exp,
            'contrastive_exp': args.contrastive_exp,
            'contrastive_idt': args.contrastive_idt,
            'stylegan_weight': args.stylegan_weight,
        }
        self.init_networks(args, training)

        if training:
            self.init_losses(args)

    def init_networks(self, args, training):
        self.use_seg = args.use_seg
        self.separate_idt = args.separate_idt
        self.local_encoder = volumetric_avatar.LocalEncoder(
            use_amp_autocast=args.use_amp_autocast,
            gen_upsampling_type=args.gen_upsampling_type,
            gen_downsampling_type=args.gen_downsampling_type,
            gen_input_image_size=args.image_size,
            gen_latent_texture_size=args.gen_latent_texture_size,
            gen_latent_texture_depth=args.gen_latent_texture_depth,
            warp_norm_grad=args.warp_norm_grad,
            gen_num_channels=args.gen_num_channels,
            enc_channel_mult=args.enc_channel_mult,
            norm_layer_type=args.norm_layer_type,
            num_gpus=args.num_gpus,
            gen_max_channels=args.gen_max_channels_loc_enc,
            enc_block_type=args.enc_block_type,
            gen_activation_type=args.gen_activation_type,
            gen_latent_texture_channels=args.gen_latent_texture_channels,
            in_channels=3
        )

        if self.use_seg and self.args.use_back:
            self.local_encoder_seg = volumetric_avatar.LocalEncoderSeg(
                use_amp_autocast=args.use_amp_autocast,
                gen_upsampling_type=args.gen_upsampling_type,
                gen_downsampling_type=args.gen_downsampling_type,
                gen_num_channels=args.gen_num_channels,
                enc_channel_mult=args.enc_channel_mult,
                norm_layer_type=args.norm_layer_type,
                num_gpus=args.num_gpus,
                gen_input_image_size=args.image_size,
                gen_latent_texture_size=args.gen_latent_texture_size,
                gen_max_channels=args.gen_max_channels,
                enc_block_type=args.enc_block_type,
                gen_activation_type=args.gen_activation_type,
                seg_out_channels=self.background_net_input_channels,
                in_channels=3
            )

        # self.local_encoder_mask = volumetric_avatar.LocalEncoderMask(
        #     use_amp_autocast=args.use_amp_autocast,
        #     gen_upsampling_type=args.gen_upsampling_type,
        #     gen_downsampling_type=args.gen_downsampling_type,
        #     gen_num_channels=args.gen_num_channels,
        #     enc_channel_mult=args.enc_channel_mult,
        #     norm_layer_type=args.norm_layer_type,
        #     num_gpus=args.num_gpus,
        #     gen_input_image_size=args.image_size,
        #     gen_latent_texture_size=32,
        #     gen_max_channels=160,
        #     enc_block_type=args.enc_block_type,
        #     gen_activation_type=args.gen_activation_type,
        #     seg_out_channels=64,
        #     in_channels=5
        # )

        if self.args.volume_rendering:
            self.volume_renderer = volumetric_avatar.VolumeRenderer(dec_channels= 1024,
                                                                    img_channels=args.dec_max_channels,
                                                                    squeeze_dim=args.squeeze_dim,
                                                                    features_sigm=args.features_sigm,
                                                                    depth_resolution=args.coarse_num_sample,
                                                                    hidden_vol_dec_dim=args.hidden_vol_dec_dim,
                                                                    num_layers_vol_dec=args.num_layers_vol_dec)
        else:
            self.volume_renderer = nn.Linear(1,1)

        self.idt_embedder = volumetric_avatar.IdtEmbed(
            idt_backbone=args.idt_backbone,
            use_amp_autocast=args.use_amp_autocast,
            num_source_frames=args.num_source_frames,
            idt_output_size=args.idt_output_size,
            idt_output_channels=args.idt_output_channels,
            num_gpus=args.num_gpus,
            norm_layer_type=args.norm_layer_type,
            idt_image_size=args.idt_image_size)

        if self.separate_idt:
            self.idt_embedder_face = volumetric_avatar.IdtEmbed(
                idt_backbone=args.idt_backbone,
                use_amp_autocast=args.use_amp_autocast,
                num_source_frames=args.num_source_frames,
                idt_output_size=args.idt_output_size,
                idt_output_channels=args.idt_output_channels,
                num_gpus=args.num_gpus,
                norm_layer_type=args.norm_layer_type,
                idt_image_size=args.idt_image_size)

        self.expression_embedder = volumetric_avatar.ExpressionEmbed(
            use_amp_autocast=args.use_amp_autocast,
            lpe_head_backbone=args.lpe_head_backbone,
            lpe_face_backbone=args.lpe_face_backbone,
            image_size=args.exp_image_size,
            project_dir=args.project_dir,
            num_gpus=args.num_gpus,
            lpe_output_channels=args.lpe_output_channels,
            lpe_final_pooling_type=args.lpe_final_pooling_type,
            lpe_output_size=args.lpe_output_size,
            lpe_head_transform_sep_scales=args.lpe_head_transform_sep_scales,
            norm_layer_type=args.norm_layer_type)

        self.xy_generator = volumetric_avatar.WarpGenerator(
            eps=args.eps,
            num_gpus=args.num_gpus,
            use_amp_autocast=args.use_amp_autocast,
            gen_adaptive_conv_type=args.gen_adaptive_conv_type,
            gen_activation_type=args.gen_activation_type,
            gen_upsampling_type=args.gen_upsampling_type,
            gen_downsampling_type=args.gen_downsampling_type,
            gen_dummy_input_size=args.gen_embed_size,
            gen_latent_texture_depth=args.gen_latent_texture_depth,
            gen_latent_texture_size=args.gen_latent_texture_size,
            gen_max_channels=args.gen_max_channels,
            gen_num_channels=args.gen_num_channels,
            gen_use_adaconv=args.gen_use_adaconv,
            gen_adaptive_kernel=args.gen_adaptive_kernel,
            gen_embed_size=args.gen_embed_size,
            warp_output_size=args.warp_output_size,
            warp_channel_mult=args.warp_channel_mult,
            warp_block_type=args.warp_block_type,
            norm_layer_type=args.norm_layer_type,
            input_channels=args.gen_max_channels)

        self.uv_generator = volumetric_avatar.WarpGenerator(
            eps=args.eps,
            num_gpus=args.num_gpus,
            use_amp_autocast=args.use_amp_autocast,
            gen_adaptive_conv_type=args.gen_adaptive_conv_type,
            gen_activation_type=args.gen_activation_type,
            gen_upsampling_type=args.gen_upsampling_type,
            gen_downsampling_type=args.gen_downsampling_type,
            gen_dummy_input_size=args.gen_embed_size,
            gen_latent_texture_depth=args.gen_latent_texture_depth,
            gen_latent_texture_size=args.gen_latent_texture_size,
            gen_max_channels=args.gen_max_channels,
            gen_num_channels=args.gen_num_channels,
            gen_use_adaconv=args.gen_use_adaconv,
            gen_adaptive_kernel=args.gen_adaptive_kernel,
            gen_embed_size=args.gen_embed_size,
            warp_output_size=args.warp_output_size,
            warp_channel_mult=args.warp_channel_mult,
            warp_block_type=args.warp_block_type,
            norm_layer_type=args.norm_layer_type,
            input_channels=args.gen_max_channels)


        m=2 if self.args.cat_em else 1
        self.warp_embed_head_orig = nn.Conv2d(  # это + на схеме, который объединяет эмбеддинги
            in_channels=args.gen_max_channels*m,
            out_channels=args.gen_max_channels,
            kernel_size=(1, 1),
            bias=False)

        if self.args.less_em:
            self.pose_embed_code = nn.Conv2d(
                in_channels=args.gen_max_channels,
                out_channels=args.gen_max_channels//args.reduce_em_ratio,
                kernel_size=(1, 1),
                bias=False)

            self.pose_embed_decode = nn.Conv2d(
                in_channels=args.gen_max_channels//args.reduce_em_ratio,
                out_channels=args.gen_max_channels,
                kernel_size=(1, 1),
                bias=False)

        # self.warp_idt_s = nn.Conv2d(  # это + на схеме, который объединяет эмбеддинги
        #     in_channels=args.gen_max_channels,
        #     out_channels=args.gen_max_channels,
        #     kernel_size=(1, 1),
        #     bias=False)
        #
        #
        # self.warp_idt_d = nn.Conv2d(  # это + на схеме, который объединяет эмбеддинги
        #     in_channels=args.gen_max_channels,
        #     out_channels=args.gen_max_channels,
        #     kernel_size=(1, 1),
        #     bias=False)



        if self.args.use_tensor:
            d = 16
            s = 64
            c = 96
            self.avarage_tensor = nn.Parameter(torch.zeros((1,c,d,s,s)).uniform_(-1, 1)*math.sqrt(6./(d*s*s*c)), requires_grad = True)

        self.volume_process_net = volumetric_avatar.Unet3D(
            eps=args.eps,
            num_gpus=args.num_gpus,
            gen_embed_size=args.gen_embed_size,
            gen_adaptive_kernel=args.gen_adaptive_kernel,
            use_amp_autocast=args.use_amp_autocast,
            gen_use_adanorm=args.gen_use_adanorm,
            gen_use_adaconv=args.gen_use_adaconv,
            gen_upsampling_type=args.gen_upsampling_type,
            gen_downsampling_type=args.gen_downsampling_type,
            gen_dummy_input_size=args.gen_dummy_input_size,
            gen_latent_texture_size=args.gen_latent_texture_size,
            gen_latent_texture_depth=args.gen_latent_texture_depth,
            gen_adaptive_conv_type=args.gen_adaptive_conv_type,
            gen_latent_texture_channels=args.gen_latent_texture_channels,
            gen_activation_type=args.gen_activation_type,
            gen_max_channels=args.gen_max_channels_unet3d,
            warp_norm_grad=args.warp_norm_grad,
            warp_block_type=args.warp_block_type,
            tex_pred_rgb=args.tex_pred_rgb,
            image_size=args.image_size,
            tex_use_skip_resblock=args.tex_use_skip_resblock,
            norm_layer_type=args.norm_layer_type,
        )

        if self.args.source_volume_num_blocks>0:
            self.volume_source_net = volumetric_avatar.ResBlocks3d(
                num_gpus=args.num_gpus,
                norm_layer_type=args.norm_layer_type,
                input_channels=self.args.latent_volume_channels,
                num_blocks=self.args.source_volume_num_blocks,
                activation_type=args.gen_activation_type,
                conv_layer_type='conv_3d',
                channels=None,
                # channels=[self.args.latent_volume_channels, 2*self.args.latent_volume_channels, 4*self.args.latent_volume_channels, 4*self.args.latent_volume_channels, 2*self.args.latent_volume_channels, self.args.latent_volume_channels],
                num_layers=3,

            )

        if self.args.pred_volume_num_blocks>0:
            self.volume_pred_net = volumetric_avatar.ResBlocks3d(
                num_gpus=args.num_gpus,
                norm_layer_type=args.norm_layer_type,
                input_channels=self.args.latent_volume_channels,
                num_blocks=self.args.pred_volume_num_blocks,
                activation_type=args.gen_activation_type,
                conv_layer_type='conv_3d',
                channels=None,
                # channels=[self.args.latent_volume_channels, 2*self.args.latent_volume_channels, 4*self.args.latent_volume_channels, 4*self.args.latent_volume_channels, 2*self.args.latent_volume_channels, self.args.latent_volume_channels],
                num_layers=3,

            )


        self.decoder = volumetric_avatar.Decoder(
            eps=args.eps,
            image_size=args.image_size,
            use_amp_autocast=args.use_amp_autocast,
            gen_embed_size=args.gen_embed_size,
            gen_adaptive_kernel=args.gen_adaptive_kernel,
            gen_adaptive_conv_type=args.gen_adaptive_conv_type,
            gen_latent_texture_size=args.gen_latent_texture_size,
            in_channels=args.gen_latent_texture_channels * args.gen_latent_texture_depth,
            gen_num_channels=args.gen_num_channels,
            dec_max_channels=args.dec_max_channels,
            gen_use_adanorm=False,
            gen_activation_type=args.gen_activation_type,
            gen_use_adaconv=args.gen_use_adaconv,
            dec_channel_mult=args.dec_channel_mult,
            dec_num_blocks=args.dec_num_blocks,
            dec_up_block_type=args.dec_up_block_type,
            dec_pred_seg=args.dec_pred_seg,
            dec_seg_channel_mult=args.dec_seg_channel_mult,
            dec_pred_conf=args.dec_pred_conf,
            dec_conf_ms_names=args.dec_conf_ms_names,
            dec_conf_names=args.dec_conf_names,
            dec_conf_ms_scales=args.dec_conf_ms_scales,
            dec_conf_channel_mult=args.dec_conf_channel_mult,
            gen_downsampling_type=args.gen_downsampling_type,
            num_gpus=args.num_gpus,
            norm_layer_type=args.norm_layer_type,
            bigger=self.args.dec_bigger,
            vol_render=args.volume_rendering)

        if args.landmarks:
            self.retinaface = Retinaface.Retinaface()
        if args.warp_norm_grad:
            self.grid_sample = volumetric_avatar.GridSample(args.gen_latent_texture_size)
        else:
            self.grid_sample = lambda inputs, grid: F.grid_sample(inputs.float(), grid.float(),
                                                                  padding_mode=self.args.grid_sample_padding_mode)

        if args.cycle_idn or args.cycle_exp or self.pred_seg:
            self.get_mask = MODNET()

        self.only_cycle_embed = args.only_cycle_embed

        self.use_masked_aug = args.use_masked_aug
        self.num_b_negs = self.args.num_b_negs
        print(f'Segmentation: use_seg {self.use_seg}, use_masked_aug {self.use_masked_aug}')

        if self.use_seg and self.args.use_back:
            in_u = self.background_net_input_channels
            c = self.args.latent_volume_channels
            d = self.args.latent_volume_depth
            self.background_net_out_channels = self.args.latent_volume_depth * self.args.latent_volume_channels
            # self.background_net_out_channels = 1024
            u = self.background_net_out_channels

            self.backgroung_adding = nn.Sequential(*[nn.Conv2d(
                in_channels=c * d + u,
                out_channels=c * d,
                kernel_size=(1, 1),
                padding=0,
                bias=False),
                nn.ReLU(),
            ])

            self.background_process_net = volumetric_avatar.UNet(in_u, u, base=64, max_ch=1024, norm='gn')
        self.prev_targets = None
        self.autocast = args.use_amp_autocast
        self.apply(weight_init.weight_init(args.init_type, args.init_gain))
        self.dec_pred_conf = args.dec_pred_conf
        self.sep_train_losses = args.sep_train_losses
        self.resize_warp = args.warp_output_size != args.gen_latent_texture_size
        self.warp_resize_stride = (
            1, args.warp_output_size // args.gen_latent_texture_size,
            args.warp_output_size // args.gen_latent_texture_size)
        self.resize_func = lambda x: F.avg_pool3d(x, kernel_size=self.warp_resize_stride,
                                                  stride=self.warp_resize_stride)
        self.resize_warp_func = lambda x: F.avg_pool3d(x.permute(0, 4, 1, 2, 3), kernel_size=self.warp_resize_stride,
                                                       stride=self.warp_resize_stride).permute(0, 2, 3, 4, 1)

        self.pose_unsqueeze = nn.Linear(args.lpe_output_channels, args.gen_max_channels * self.embed_size ** 2,
                                        bias=False)  # embedding to warping

        for net_name in ['local_encoder', 'local_encoder_seg', 'local_encoder_mask', 'idt_embedder',
                         'expression_embedder',
                         'xy_generator', 'uv_generator', 'warp_embed_head_orig', 'volume_process_net', 'volume_source_net', 'volume_pred_net',
                         'decoder', 'backgroung_adding', 'background_process_net']:
            try:
                net = getattr(self, net_name)
                pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print(f'Number of trainable params in {net_name} : {pytorch_total_params}')
            except Exception as e:
                print(e)

        if training:
            self.discriminator = basic_avatar.MultiScaleDiscriminator(
                min_channels=args.dis_num_channels,
                max_channels=args.dis_max_channels,
                num_blocks=args.dis_num_blocks,
                input_channels=3,
                input_size=args.image_size,
                num_scales=args.dis_num_scales)

            self.discriminator.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))

            # if self.args.use_hq_disc:
            #     self.discriminator2 = basic_avatar.MultiScaleDiscriminator(
            #         min_channels=args.dis2_num_channels,
            #         max_channels=args.dis2_max_channels,
            #         num_blocks=args.dis2_num_blocks,
            #         input_channels=3,
            #         input_size=args.image_size,
            #         num_scales=args.dis2_num_scales)
            #
            #     self.discriminator2.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))

            pytorch_total_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
            print(f'Number of trainable params in discriminator : {pytorch_total_params}')

            if self.use_stylegan_d:
                self.r1_loss = torch.tensor(0.0)
                self.stylegan_discriminator = basic_avatar.DiscriminatorStyleGAN2(size=self.args.image_size,
                                                                                  channel_multiplier=1, my_ch=2)
                # self.stylegan_discriminator.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))
                pytorch_total_params = sum(
                    p.numel() for p in self.stylegan_discriminator.parameters() if p.requires_grad)
                print(f'Number of trainable params in stylegan2 discriminator : {pytorch_total_params}')

        if self.args.estimate_head_pose_from_keypoints:
            self.head_pose_regressor = volumetric_avatar.HeadPoseRegressor(args.head_pose_regressor_path, args.num_gpus)

        # if self.separate_idt:
        self.face_idt = volumetric_avatar.FaceParsing(None, 'cuda')

        self.get_face_vector = volumetric_avatar.utils.Face_vector(self.head_pose_regressor, half=False)
        self.get_face_vector_resnet = volumetric_avatar.utils.Face_vector_resnet(half=False)

        grid_s = torch.linspace(-1, 1, self.args.aug_warp_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('identity_grid_2d', torch.stack([u, v], dim=2).view(1, -1, 2), persistent=False)

        grid_s = torch.linspace(-1, 1, self.args.latent_volume_size)
        grid_z = torch.linspace(-1, 1, self.args.latent_volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer('identity_grid_3d', torch.stack([u, v, w, e], dim=3).view(1, -1, 4), persistent=False)

        # Apply spectral norm
        if args.use_sn:
            # print('Applying SN')
            spn_layers = args_utils.parse_str_to_list(args.spn_layers, sep=',')
            spn_nets_names = args_utils.parse_str_to_list(args.spn_networks, sep=',')

            for net_name in spn_nets_names:
                try:
                    net = getattr(self, net_name)
                    # print('apply SN to: ', net_name)
                    net.apply(lambda module: spectral_norm.apply_spectral_norm(module, apply_to=spn_layers))

                except Exception as e:
                    print(e)
                    print(f'there is no {net_name} network')

        if args.use_ws:
            print('Applying WS')
            ws_nets_names = args_utils.parse_str_to_list(args.ws_networks, sep=',')
            for net_name in ws_nets_names:
                try:
                    net = getattr(self, net_name)
                    print('apply ws to: ', net_name)
                    new_net = volumetric_avatar.utils.replace_conv_to_ws_conv(net, conv2d=True, conv3d=True)
                    setattr(self, net_name, new_net)
                    # net.apply(lambda module: volumetric_avatar.utils.replace_conv_to_ws_conv(module))
                except Exception as e:
                    print(e)
                    print(f'there is no {net_name} network')

    def init_losses(self, args):
        return init_losses(self, args)

    @torch.no_grad()
    def get_face_warp(self, grid, params_ffhq):
        grid = grid.view(grid.shape[0], -1, 2)
        face_warp = point_transforms.align_ffhq_with_zoom(grid, params_ffhq)
        face_warp = face_warp.view(face_warp.shape[0], self.args.aug_warp_size, self.args.aug_warp_size, 2)

        return face_warp

    def _forward(self, data_dict, visualize, ffhq_per_b=0, iteration=0):
        self.visualize = visualize
        b = data_dict['source_img'].shape[0]


        if self.args.use_mix_mask:
            face_mask_source, _, _, cloth_s = self.face_idt.forward(data_dict['source_img'])
            face_mask_target, _, _, cloth_t = self.face_idt.forward(data_dict['target_img'])
            trashhold = 0.6
            face_mask_source = (face_mask_source > trashhold).float()
            face_mask_target = (face_mask_target > trashhold).float()

            data_dict['source_mask_modnet'] = data_dict['source_mask']
            data_dict['target_mask_modnet'] = data_dict['target_mask']
            data_dict['source_mask_face_pars'] = (face_mask_source + cloth_s).float()
            data_dict['target_mask_face_pars'] = (face_mask_target + cloth_t).float()

            trashhold = 0.8
            # data_dict['pre_ready_mask_sou'] = ((data_dict['source_mask']*face_mask_source)>trashhold).float()
            # data_dict['pre_ready_mask_tar'] = ((data_dict['target_mask']*face_mask_target)>trashhold).float()
            data_dict['pre_ready_mask_sou'] = ((data_dict['source_mask']) > trashhold).float()
            data_dict['pre_ready_mask_tar'] = ((data_dict['target_mask']) > trashhold).float()
            aa = data_dict['pre_ready_mask_sou'].cpu().squeeze().numpy()
            bb = data_dict['pre_ready_mask_tar'].cpu().squeeze().numpy()
            labels_source = label(aa)
            labels_target = label(bb)

            # ious_s = np.array([self.iou_numpy(aa, labels_source == i) for i in range(1, 6)])
            # ious_t = np.array([self.iou_numpy(bb, labels_target == i) for i in range(1, 6)])

            ious_s = np.array([np.sum(labels_source == i, axis=(1, 2)) for i in range(1, 6)])
            ious_t = np.array([np.sum(labels_target == i, axis=(1, 2)) for i in range(1, 6)])

            index_source = np.argmax(ious_s, axis=0)
            index_target = np.argmax(ious_t, axis=0)

            lab_map_source = torch.tensor((labels_source == 1 + np.expand_dims(index_source, axis=(1, 2)))).to(
                data_dict['pre_ready_mask_sou'].device)
            lab_map_target = torch.tensor((labels_target == 1 + np.expand_dims(index_target, axis=(1, 2)))).to(
                data_dict['pre_ready_mask_tar'].device)

            data_dict['source_mask_s'] = (data_dict['source_mask_modnet'] * cloth_s > trashhold).float()
            data_dict['target_mask_t'] = (data_dict['target_mask_modnet'] * cloth_t > trashhold).float()
            # data_dict['source_mask'] = (data_dict['source_mask']*face_mask_source*lab_map_source.unsqueeze(dim=1)).float()
            # data_dict['target_mask'] = (data_dict['target_mask']*face_mask_target*lab_map_target.unsqueeze(dim=1)).float()
            data_dict['source_mask'] = (data_dict['source_mask'] * lab_map_source.unsqueeze(dim=1)).float()
            data_dict['target_mask'] = (data_dict['target_mask'] * lab_map_target.unsqueeze(dim=1)).float()

        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth

        # Estimate head rotation and corresponded warping
        if self.args.estimate_head_pose_from_keypoints:
            with torch.no_grad():
                data_dict['source_theta'] = self.head_pose_regressor.forward(data_dict['source_img'])
                data_dict['target_theta'] = self.head_pose_regressor.forward(data_dict['target_img'])

            grid = self.identity_grid_3d.repeat_interleave(b, dim=0)

            inv_source_theta = data_dict['source_theta'].float().inverse().type(data_dict['source_theta'].type())
            data_dict['source_rotation_warp'] = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
            data_dict['target_rotation_warp'] = grid.bmm(data_dict['target_theta'][:, :3].transpose(1, 2)).view(-1, d,
                                                                                                                s, s, 3)

        else:
            data_dict['source_rotation_warp'] = point_transforms.world_to_camera(self.identity_grid_3d[..., :3],
                                                                                 data_dict['source_params_3dmm']).view(
                b, d, s, s, 3)
            data_dict['target_rotation_warp'] = point_transforms.camera_to_world(self.identity_grid_3d[..., :3],
                                                                                 data_dict['target_params_3dmm']).view(
                b, d, s, s, 3)

        if self.use_seg:
            if self.separate_idt:
                _, face_mask, body_mask, _ = self.face_idt.forward(data_dict['source_img'])
                # data_dict['source_mask'] = body_mask*0.5 + face_mask
                body_mask = data_dict['source_mask'] * (1 - face_mask)
                data_dict['idt_embed'] = self.idt_embedder(data_dict['source_img'] * body_mask)
                data_dict['idt_embed_face'] = self.idt_embedder_face(data_dict['source_img'] * face_mask)
                # data_dict['idt_embed']+=self.warp_idt_s(data_dict['idt_embed_face'])
                target_latent_volume = self.local_encoder(data_dict['source_img'] * data_dict['source_mask'])
                # source_latents = self.local_encoder(torch.cat((data_dict['source_img'] * data_dict['source_mask'], data_dict['source_mask']), dim=1))
                # source_latents = self.local_encoder(torch.cat((data_dict['source_img'] * data_dict['source_mask'], face_mask_source * data_dict['source_mask'], data_dict['source_mask']), dim=1))
            else:
                data_dict['idt_embed'] = self.idt_embedder(data_dict['source_img'] * data_dict['source_mask'])
                # with torch.no_grad():
                target_latent_volume = self.local_encoder(data_dict['source_img'] * data_dict['source_mask'])
                # source_latents = self.local_encoder(torch.cat((data_dict['source_img'] * data_dict['source_mask'], data_dict['source_mask']), dim=1))
                # source_latents = self.local_encoder(torch.cat((data_dict['source_img'] * data_dict['source_mask'], face_mask_source*data_dict['source_mask'], data_dict['source_mask']), dim=1))

            # data_dict['idt_embed'] = self.idt_embedder(data_dict['source_img'])
            # source_latents = self.local_encoder(data_dict['source_img'] )

        else:
            if self.separate_idt:
                raise ValueError('separate_idt without segmentation')
            else:
                data_dict['idt_embed'] = self.idt_embedder(data_dict['source_img'])
                target_latent_volume = self.local_encoder(data_dict['source_img'])



        data_dict = self.expression_embedder(data_dict, self.args.estimate_head_pose_from_keypoints,
                                             self.use_masked_aug)

        # Produce embeddings for warping
        source_warp_embed_dict, target_warp_embed_dict, mixing_warp_embed_dict, embed_dict = self.predict_embed(
            data_dict)

        # Predict a warping from image features to texture inputs
        xy_gen_outputs = self.xy_generator(source_warp_embed_dict)
        data_dict['source_delta_xy'] = xy_gen_outputs[0]

        # Predict warping to the texture coords space
        source_xy_warp = xy_gen_outputs[0]
        source_xy_warp_resize = source_xy_warp
        if self.resize_warp:
            source_xy_warp_resize = self.resize_warp_func(
                source_xy_warp_resize)  # if warp is predicted at higher resolution than resolution of texture

        # Predict warping of texture according to target pose
        target_uv_warp, data_dict['target_delta_uv'] = self.uv_generator(target_warp_embed_dict)
        target_uv_warp_resize = target_uv_warp
        if self.resize_warp:
            target_uv_warp_resize = self.resize_warp_func(target_uv_warp_resize)

        data_dict['source_xy_warp_resize'] = source_xy_warp_resize
        data_dict['target_uv_warp_resize'] = target_uv_warp_resize

        if self.use_seg and self.args.use_back:
            seg_encod_input = data_dict['source_img'] * (1 - data_dict['source_mask'])

            # data_dict['target_mask'] = ((data_dict['target_mask'] + data_dict['target_mask_s'])>0.5).float()
            # seg_encod_input = torch.cat((data_dict['source_img'] * (data_dict['source_mask_s']), data_dict['target_mask_s']), dim=1)
            source_latents_background = self.local_encoder_seg(seg_encod_input)


            target_latent_feats_back = self.background_process_net(source_latents_background)


        else:
            pass

        # Reshape latents into 3D volume
        target_latent_volume = target_latent_volume.view(b, c, d, s, s)

        # Warp latent texture from source
        # if ffhq_per_b>0:
        #     data_dict['source_rotation_warp'] = torch.cat((data_dict['source_rotation_warp'][:-ffhq_per_b], data_dict['source_rotation_warp'][-ffhq_per_b:].detach()))
        #     data_dict['source_xy_warp_resize'] = torch.cat((data_dict['source_xy_warp_resize'][:-ffhq_per_b], data_dict['source_xy_warp_resize'][-ffhq_per_b:].detach()))

        if self.args.use_tensor:
            target_latent_volume+=self.avarage_tensor

        if self.args.source_volume_num_blocks > 0:
            target_latent_volume = self.volume_source_net(target_latent_volume)

        target_latent_volume = self.grid_sample(
            self.grid_sample(target_latent_volume, data_dict['source_rotation_warp']),
            data_dict['source_xy_warp_resize'])

        # target_latent_volume = self.grid_sample(target_latent_volume, data_dict['source_rotation_warp'])

        # Process latent texture
        target_latent_volume = self.volume_process_net(target_latent_volume, embed_dict)



        if self.args.pred_volume_num_blocks > 0:
            target_latent_volume = self.volume_pred_net(target_latent_volume)
        # Warp latent texture to target
        # if ffhq_per_b>0:
        #     data_dict['target_uv_warp_resize'] = torch.cat((data_dict['target_uv_warp_resize'][:-ffhq_per_b], data_dict['target_uv_warp_resize'][-ffhq_per_b:].detach()))
        #     data_dict['target_rotation_warp'] = torch.cat((data_dict['target_rotation_warp'][:-ffhq_per_b], data_dict['target_rotation_warp'][-ffhq_per_b:].detach()))

        aligned_target_volume = self.grid_sample(
            self.grid_sample(target_latent_volume, data_dict['target_uv_warp_resize']),
            data_dict['target_rotation_warp'])

        # if ffhq_per_b>0:
        #     # aligned_target_volume = torch.cat((aligned_target_volume[:-ffhq_per_b], aligned_target_volume[-ffhq_per_b:].detach()))
        #     aligned_target_volume = aligned_target_volume.detach()

        aligned_target_volume_flip = None

        if self.use_seg and self.args.use_back:
            aligned_target_volume = aligned_target_volume.view(b, c * d, s, s)
            aligned_target_volume = self.backgroung_adding(
                torch.cat((aligned_target_volume, target_latent_feats_back), dim=1))
            # target_latent_feats = target_latent_feats + target_latent_feats_back
        else:
            if self.args.volume_rendering:
                aligned_target_volume, data_dict['pred_tar_img_vol'], data_dict['pred_tar_depth_vol'] = self.volume_renderer(aligned_target_volume)
                # print('sssssssssssssssssssssssssssss', aligned_target_volume.shape)
            else:
                aligned_target_volume = aligned_target_volume.view(b, c * d, s, s)

        img, seg = self.decoder(data_dict, embed_dict, aligned_target_volume, aligned_target_volume_flip is not None)

        # Decode into image
        data_dict['pred_target_img'] = img[:b]
        # if self.pred_seg:
        # data_dict['pred_target_seg'] = seg[:b]
        if not self.args.use_back:
            data_dict['pred_target_img'] = data_dict[
                'pred_target_img']  # data_dict['target_mask'].detach() #data_dict['pred_target_seg'] #TODO change to pred_seg

            data_dict['target_img'] = data_dict['target_img'] * data_dict['target_mask'].detach()
            if self.args.green:
                green = torch.ones_like(data_dict['target_img'])*(1-data_dict['target_mask'].detach())
                green[:, 0, :, :] = 0
                green[:, 2, :, :] = 0
                data_dict['target_img'] += green

        if self.training and self.pred_flip:
            data_dict['pred_target_img_flip'] = img[b:2 * b]

        # Cross-reenactment
        if self.pred_mixing or not self.training:
            if (self.weights['cycle_idn'] or self.weights['cycle_exp']) and self.training:

                #######################################################################################################
                # Mixing prediction

                mixing_uv_warp, data_dict['target_delta_uv'] = self.uv_generator(mixing_warp_embed_dict)
                mixing_uv_warp_resize = mixing_uv_warp
                if self.resize_warp:
                    mixing_uv_warp_resize = self.resize_warp_func(mixing_uv_warp_resize)

                aligned_mixing_feat = self.grid_sample(target_latent_volume, mixing_uv_warp_resize)

                mixing_theta = self.get_mixing_theta(data_dict['source_theta'], data_dict['target_theta'])
                mixing_align_warp = self.identity_grid_3d.repeat_interleave(b, dim=0)
                mixing_align_warp = mixing_align_warp.bmm(mixing_theta.transpose(1, 2)).view(b,
                                                                                             *mixing_uv_warp.shape[1:4],
                                                                                             3)
                if self.resize_warp:
                    mixing_align_warp_resize = self.resize_warp_func(mixing_align_warp)
                else:
                    mixing_align_warp_resize = mixing_align_warp

                aligned_mixing_feat = self.grid_sample(aligned_mixing_feat, mixing_align_warp_resize)

                if self.args.volume_rendering:
                    aligned_mixing_feat, data_dict['pred_mixing_img_vol'], data_dict['pred_mixing_depth_vol'] = self.volume_renderer(aligned_mixing_feat)

                else:
                    aligned_mixing_feat = aligned_mixing_feat.view(b, c * d, s, s)

                self.decoder.train()

                if self.use_seg and self.args.use_back:
                    # aligned_mixing_feat = aligned_mixing_feat + target_latent_feats_back.detach()
                    aligned_mixing_feat = self.backgroung_adding(
                        torch.cat((aligned_mixing_feat, target_latent_feats_back.detach()), dim=1))

                # aligned_mixing_feat = torch.cat((aligned_mixing_feat, source_mask), dim=1)

                data_dict['pred_mixing_img'], pred_mixing_seg = self.decoder(data_dict, embed_dict,
                                                                             aligned_mixing_feat, False)[:2]

                mask_mix_ = self.get_mask.forward(data_dict['pred_mixing_img'])
                # if self.pred_seg:
                # data_dict['pred_mixing_seg'] = pred_mixing_seg
                data_dict['pred_mixing_img'] = data_dict['pred_mixing_img']  # * mask_mix.detach()

                # resize and save matte
                _, mask_mix, body_mix_mask, _ = self.face_idt.forward(data_dict['pred_mixing_img'])
                body_mix_mask = mask_mix_ * (1 - mask_mix)
                mask_mix = F.interpolate(mask_mix.float(), size=(
                data_dict['pred_mixing_img'].shape[2], data_dict['pred_mixing_img'].shape[3]), mode='area')
                body_mix_mask = F.interpolate(body_mix_mask.float(), size=(
                data_dict['pred_mixing_img'].shape[2], data_dict['pred_mixing_img'].shape[3]), mode='area')

                data_dict['pred_mixing_masked_img'] = data_dict['pred_mixing_img'] * mask_mix_
                data_dict['pred_mixing_mask'] = mask_mix_
                # data_dict['pred_mixing_mask'] = mask_mix + body_mix_mask * 0.5

                ########################################################################################################
                # Identity contrastive
                if self.args.separate_idt:
                    data_dict_idn = copy.copy(data_dict)
                    with torch.no_grad():
                        data_dict_idn['target_theta'] = self.head_pose_regressor.forward(data_dict['pred_mixing_img'])
                    idt_embed_true = data_dict_idn['idt_embed']
                    ######
                    idt_embed_face_mix = self.idt_embedder_face(data_dict['pred_mixing_img'] * mask_mix)
                    idt_embed_mix = self.idt_embedder(data_dict['pred_mixing_img'] * body_mix_mask)

                    # self.warp_idt_d.eval()
                    # self.warp_idt_s.eval()

                    # data_dict['idt_embed_face_mix'] = self.warp_idt_d(idt_embed_face_mix)
                    data_dict['idt_embed_face_mix'] = idt_embed_face_mix
                    data_dict_idn['idt_embed'] = idt_embed_face_mix + idt_embed_mix

                    _, mask, body_mask, _ = self.face_idt.forward(data_dict['pred_target_img'])
                    idt_embed_cycle_face = self.idt_embedder_face(data_dict['pred_target_img'] * mask)
                    # data_dict['idt_embed_face_pred'] = self.warp_idt_d(idt_embed_cycle_face)
                    data_dict['idt_embed_face_pred'] = idt_embed_cycle_face

                    _, mask, body_mask, _ = self.face_idt.forward(data_dict['target_img'])
                    idt_embed_cycle_face = self.idt_embedder_face(data_dict['target_img'] * mask)
                    # data_dict['idt_embed_face_target'] = self.warp_idt_d(idt_embed_cycle_face)
                    data_dict['idt_embed_face_target'] = idt_embed_cycle_face

                    # data_dict['idt_embed_face'] = self.warp_idt_d(data_dict['idt_embed_face'])

                    # self.warp_idt_d.train()
                    # self.warp_idt_s.train()
                # _, target_warp_cycle_idn, _, origin_cycle_int = self.predict_embed(data_dict_idn)
                #
                # ############################
                # # Predict cycle (if needed)
                # if not self.only_cycle_embed:
                #     target_uv_warp_idn, data_dict_idn['target_delta_uv'] = self.uv_generator(target_warp_cycle_idn)
                #     target_uv_warp_resize_idn = target_uv_warp_idn
                #     if self.resize_warp:
                #         target_uv_warp_resize_idn = self.resize_warp_func(target_uv_warp_resize)
                #
                #     pred_cycle_identical_vol = \
                #         self.grid_sample(self.grid_sample(target_latent_volume, target_uv_warp_resize_idn),
                #                          data_dict['target_rotation_warp'])
                #
                #     if self.use_seg and self.args.use_back:
                #         target_latent_feats_idn = pred_cycle_identical_vol.view(b, c * d, s, s)
                #         target_latent_feats_idn = target_latent_feats_idn + target_latent_feats_back.detach()
                #     else:
                #         target_latent_feats_idn = pred_cycle_identical_vol.view(b, c * d, s, s)
                #
                #     identical_cycle, _ = self.decoder(data_dict_idn, origin_cycle_int,
                #     target_latent_feats_idn, aligned_target_volume_flip is not None)
                #     data_dict['pred_identical_cycle'] = identical_cycle[:b]

                ########################################################################################################
                # Expression contrastive

                # data_dict_exp = copy.copy(data_dict)
                data_dict_exp = {k: torch.clone(v) if type(v) is torch.Tensor else v for k, v in data_dict.items()}
                with torch.no_grad():
                    data_dict_exp['target_theta'] = self.head_pose_regressor.forward(data_dict['pred_mixing_img'])
                    data_dict_exp['source_theta'] = self.head_pose_regressor.forward(data_dict['target_img'])
                # data_dict_idn['idt_embed'] = idt_embed_true
                data_dict_exp['rolled_mix'] = data_dict['pred_mixing_img']
                data_dict_exp['source_img'] = data_dict['pred_target_img']
                data_dict_exp['source_mask'] = data_dict['target_mask']
                data_dict_exp['target_img'] = data_dict['pred_mixing_img']
                data_dict_exp['target_mask'] = data_dict['pred_mixing_mask']
                data_dict_exp = self.expression_embedder(data_dict_exp, self.args.estimate_head_pose_from_keypoints,
                                                         self.use_masked_aug, use_aug=False)
                data_dict['mixing_img_align'] = data_dict_exp['target_img_align']

                # data_dict_exp = copy.copy(data_dict)
                data_dict_exp = {k: torch.clone(v) if type(v) is torch.Tensor else v for k, v in data_dict.items()}
                with torch.no_grad():
                    data_dict_exp['target_theta'] = self.head_pose_regressor.forward(
                        data_dict['pred_mixing_img'].roll(-1, dims=0))
                    data_dict_exp['source_theta'] = self.head_pose_regressor.forward(data_dict['target_img'])
                # data_dict_idn['idt_embed'] = idt_embed_true
                data_dict_exp['rolled_mix'] = data_dict['pred_mixing_img'].roll(-1, dims=0)
                data_dict_exp['source_img'] = data_dict['pred_target_img']
                data_dict_exp['source_mask'] = data_dict['target_mask']
                data_dict_exp['target_img'] = data_dict['pred_mixing_img'].roll(-1, dims=0)
                data_dict_exp['target_mask'] = data_dict['pred_mixing_mask'].roll(-1, dims=0)
                data_dict_exp = self.expression_embedder(data_dict_exp, self.args.estimate_head_pose_from_keypoints,
                                                         self.use_masked_aug, use_aug=False)

                _, target_warp_cycle_exp, _, origin_cycle_exp = self.predict_embed(data_dict_exp)
                data_dict['mixing_cycle_exp'] = data_dict_exp['target_pose_embed']
                data_dict['pred_cycle_exp'] = data_dict_exp['source_pose_embed']
                data_dict['rolled_mix_align'] = data_dict_exp['target_img_align']
                data_dict['rolled_mix'] = data_dict_exp['rolled_mix']

                # Predict cycle (if needed)
                # if not self.only_cycle_embed:
                #     target_uv_warp, data_dict['target_delta_uv'] = self.uv_generator(target_warp_cycle_exp)
                #     target_uv_warp_resize = target_uv_warp
                #     if self.resize_warp:
                #         target_uv_warp_resize = self.resize_warp_func(target_uv_warp_resize)
                #
                #     pred_cycle_expression_vol = \
                #         self.grid_sample(self.grid_sample(target_latent_volume, target_uv_warp_resize),
                #                          data_dict['target_rotation_warp'])
                #
                #
                #     if self.use_seg  and self.args.use_back:
                #         target_latent_feats = pred_cycle_expression_vol.view(b, c * d, s, s)
                #       #  target_latent_feats = target_latent_feats + target_latent_feats_back.detach()
                #          target_latent_feats = self.backgroung_adding(torch.cat((target_latent_feats, target_latent_feats_back.detach()), dim=1))
                #     else:
                #         target_latent_feats = pred_cycle_expression_vol.view(b, c * d, s, s)
                #     expression_cycle, _ = self.decoder(data_dict_exp, origin_cycle_exp, target_latent_feats, aligned_target_volume_flip is not None)
                #
                #     data_dict['pred_expression_cycle'] = expression_cycle[:b]
            else:
                with torch.no_grad():
                    mixing_uv_warp, data_dict['target_delta_uv'] = self.uv_generator(mixing_warp_embed_dict)
                    mixing_uv_warp_resize = mixing_uv_warp
                    if self.resize_warp:
                        mixing_uv_warp_resize = self.resize_warp_func(mixing_uv_warp_resize)

                    aligned_mixing_feat = self.grid_sample(target_latent_volume, mixing_uv_warp_resize)

                    mixing_theta = self.get_mixing_theta(data_dict['source_theta'], data_dict['target_theta'])
                    mixing_align_warp = self.identity_grid_3d.repeat_interleave(b, dim=0)
                    mixing_align_warp = mixing_align_warp.bmm(mixing_theta.transpose(1, 2)).view(b,
                                                                                                 *mixing_uv_warp.shape[
                                                                                                  1:4],
                                                                                                 3)
                    if self.resize_warp:
                        mixing_align_warp_resize = self.resize_warp_func(mixing_align_warp)
                    else:
                        mixing_align_warp_resize = mixing_align_warp

                    aligned_mixing_feat = self.grid_sample(aligned_mixing_feat, mixing_align_warp_resize)

                    if self.args.volume_rendering:
                        aligned_mixing_feat, data_dict['pred_mixing_img_vol'], data_dict[
                            'pred_mixing_depth_vol'] = self.volume_renderer(aligned_mixing_feat)
                    else:
                        aligned_mixing_feat = aligned_mixing_feat.view(b, c * d, s, s)

                    self.decoder.eval()

                    if self.use_seg and self.args.use_back:
                        # aligned_mixing_feat = aligned_mixing_feat + target_latent_feats_back.detach()
                        aligned_mixing_feat = self.backgroung_adding(
                            torch.cat((aligned_mixing_feat, target_latent_feats_back.detach()), dim=1))

                    # aligned_mixing_feat = torch.cat((aligned_mixing_feat, source_mask), dim=1)
                    data_dict['pred_mixing_img'], pred_mixing_seg = self.decoder(data_dict, embed_dict,
                                                                                 aligned_mixing_feat, False)[:2]
                    self.decoder.train()
                    # print(data_dict.keys())

                    # if self.pred_seg:
                    # data_dict['pred_mixing_mask'] = self.get_mask.forward(data_dict['pred_mixing_img'])
                    # data_dict['pred_mixing_seg'] = pred_mixing_seg.detach()
                    data_dict['pred_mixing_img'] = data_dict['pred_mixing_img'].detach()  # * data_dict['pred_mixing_seg'].detach()

        return data_dict

    def get_mixing_theta(self, source_theta, target_theta):
        source_theta = source_theta[:, :3, :]
        target_theta = target_theta[:, :3, :]
        N = self.num_source_frames
        B = source_theta.shape[0] // N
        T = target_theta.shape[0] // B

        source_theta_ = np.stack([np.eye(4) for i in range(B)])
        target_theta_ = np.stack([np.eye(4) for i in range(B * T)])


        source_theta = source_theta.view(B, N, *source_theta.shape[1:])[:, 0]  # take theta from the first source image

        if self.args.random_theta:
            r = random.randint(0, 1)
            # print(r)
            if B==2:
                self.thetas_pool.append(target_theta)
                if len(self.thetas_pool)>=50:
                    self.thetas_pool.pop(0)
                r = random.randint(0, len(self.thetas_pool)-1)
                th = self.thetas_pool[r]
                p = random.randint(0, 1)
                target_theta = target_theta if p<=0 else th
                target_theta = target_theta.view(B, T, 3, 4).roll(0, dims=0).view(B * T, 3, 4)  # shuffle target poses
            else:
                target_theta = target_theta.view(B, T, 3, 4).roll(0, dims=0).view(B * T, 3, 4)  # shuffle target poses
        else:
            target_theta = target_theta.view(B, T, 3, 4).roll(1, dims=0).view(B * T, 3, 4)  # shuffle target poses
        source_theta_[:, :3, :] = source_theta.detach().cpu().numpy()
        target_theta_[:, :3, :] = target_theta.detach().cpu().numpy()

        # Extract target translation
        target_translation = np.stack([np.eye(4) for i in range(B * T)])
        target_translation[:, :3, 3] = target_theta_[:, :3, 3]

        # Extract linear components
        source_linear_comp = source_theta_.copy()
        source_linear_comp[:, :3, 3] = 0

        target_linear_comp = target_theta_.copy()
        target_linear_comp[:, :3, 3] = 0

        pred_mixing_theta = []
        for b in range(B):
            # Sometimes the decomposition is not possible, hense try-except blocks
            try:
                source_rotation, source_stretch = linalg.polar(source_linear_comp[b])
            except:
                pred_mixing_theta += [target_theta_[b * T + t] for t in range(T)]
            else:
                for t in range(T):
                    try:
                        target_rotation, target_stretch = linalg.polar(target_linear_comp[b * T + t])
                    except:
                        pred_mixing_theta.append(source_stretch)
                    else:
                        if self.args.old_mix_pose:
                            pred_mixing_theta.append(target_translation[b * T + t] @ target_rotation @ source_stretch)
                        else:
                            pred_mixing_theta.append(source_stretch * target_stretch.mean() / source_stretch.mean() @ target_rotation @ target_translation[b * T + t])

        pred_mixing_theta = np.stack(pred_mixing_theta)

        return torch.from_numpy(pred_mixing_theta)[:, :3].type(source_theta.type()).to(source_theta.device)

    def predict_embed(self, data_dict):
        n = self.num_source_frames
        b = data_dict['source_img'].shape[0] // n
        t = data_dict['target_img'].shape[0] // b

        with amp.autocast(enabled=self.autocast):
            # Unsqueeze pose embeds for warping gen
            warp_source_embed = self.pose_unsqueeze(data_dict['source_pose_embed']).view(b * n, -1, self.embed_size,
                                                                                         self.embed_size)
            warp_target_embed = self.pose_unsqueeze(data_dict['target_pose_embed']).view(b * t, -1, self.embed_size,
                                                                                         self.embed_size)

        if self.pred_mixing:
            warp_mixing_embed = warp_target_embed.detach()
            warp_mixing_embed = warp_mixing_embed.view(b, t, -1, self.embed_size, self.embed_size).roll(1, dims=0)
            warp_mixing_embed = warp_mixing_embed.view(b * t, -1, self.embed_size, self.embed_size)

        pose_embeds = [warp_source_embed, warp_target_embed]
        # idt_embeds = [self.warp_idt_s, self.warp_idt_d]
        num_frames = [n, t]
        if self.pred_mixing:
            pose_embeds += [warp_mixing_embed]
            num_frames += [t]
            # idt_embeds+=[self.warp_idt_d]

        warp_embed_dicts = ({}, {}, {})  # source, target, mixing
        embed_dict = {}

        # Predict warp embeds
        data_dict['idt_embed'] = data_dict['idt_embed']
        for k, (pose_embed, m) in enumerate(zip(pose_embeds, num_frames)):

            if self.args.less_em:
                pose_embed = self.pose_embed_decode(self.pose_embed_code(pose_embed))


            # warp_embed_orig = self.warp_embed_head_orig(pose_embed + data_dict['idt_embed'] + idt_embeds[k](data_dict['idt_embed_face']))
            # warp_embed_orig = self.warp_embed_head_orig(pose_embed + idt_embeds[k](data_dict['idt_embed']))
            # print(pose_embed.shape, data_dict['idt_embed'].shape)
            if self.args.cat_em:
                warp_embed_orig = self.warp_embed_head_orig(torch.cat([pose_embed, data_dict['idt_embed'].repeat_interleave(m, dim=0)], dim=1))
            else:
                warp_embed_orig = self.warp_embed_head_orig((pose_embed + data_dict['idt_embed'].repeat_interleave(m, dim=0)) * 0.5)

            c = warp_embed_orig.shape[1]
            warp_embed_dicts[k]['orig'] = warp_embed_orig.view(b * m, c, self.embed_size ** 2)
            warp_embed_orig_ = warp_embed_orig.view(b * m * c, self.embed_size ** 2)

            if self.args.gen_use_adaconv:
                for name, layer in self.warp_embed_head_dict.items():
                    warp_embed_dicts[k][name] = layer(warp_embed_orig_).view(b * m, c // 2, -1)

        if self.args.gen_use_adanorm or self.args.gen_use_adaconv:
            # Predict embeds
            embed_orig = self.embed_head_orig(data_dict['idt_embed'])

            c = embed_orig.shape[1]
            embed_dict['orig'] = embed_orig.view(b, c, self.embed_size ** 2)
            embed_orig_ = embed_orig.view(b * c, self.embed_size ** 2)

        if self.args.gen_use_adaconv:
            for name, layer in self.embed_head_dict.items():
                embed_dict[name] = layer(embed_orig_).view(b, c // 2, -1)

        source_warp_embed_dict, target_warp_embed_dict, mixing_warp_embed_dict = warp_embed_dicts

        return source_warp_embed_dict, target_warp_embed_dict, mixing_warp_embed_dict, embed_dict

    def calc_train_losses(self, data_dict: dict, mode: str = 'gen', epoch=0, ffhq_per_b=0):
        return calc_train_losses(self, data_dict=data_dict, mode=mode, epoch=epoch, ffhq_per_b=ffhq_per_b)

    def calc_test_losses(self, data_dict: dict):
        return calc_test_losses(self, data_dict)

    def prepare_input_data(self, data_dict):
        return prepare_input_data(self, data_dict)

    def forward(self,
                data_dict: dict,
                phase: str = 'test',
                optimizer_idx: int = 0,
                visualize: bool = False,
                ffhq_per_b=0,
                iteration=0,
                rank=-1,
                epoch=0):
        assert phase in ['train', 'test']
        mode = self.optimizer_idx_to_mode[optimizer_idx]
        self.ffhq_per_b = ffhq_per_b
        s = self.args.image_additional_size if self.args.image_additional_size is not None else data_dict[
            'target_img'].shape[-1]
        resize = lambda img: F.interpolate(img, mode='bilinear', size=(s, s), align_corners=False)



        if mode == 'gen':
            data_dict = self.prepare_input_data(data_dict)
            # data_dict['target_img'] = self.resize_u(self.resize_d(data_dict['target_img']))
            # data_dict['source_img'] = self.resize_u(self.resize_d(data_dict['source_img']))
            if ffhq_per_b == 0:
                data_dict = self._forward(data_dict, visualize, iteration=iteration)
            else:
                data_dict = self._forward(data_dict, visualize, ffhq_per_b=ffhq_per_b, iteration=iteration)

            if phase == 'train':
                self.discriminator.eval()
                for p in self.discriminator.parameters():
                    p.requires_grad = False

                # if self.args.use_hq_disc:
                #     self.discriminator2.eval()
                #     for p in self.discriminator2.parameters():
                #         p.requires_grad = False

                with torch.no_grad():
                    # person_mask, _, _, _ = self.face_idt.forward(data_dict['target_img'])
                    _, data_dict['real_feats_gen'] = self.discriminator(resize(data_dict['target_img']))
                    # if self.args.use_hq_disc:
                    #     _, data_dict['real_feats_gen_2'] = self.discriminator2(data_dict['target_img'])

                data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(
                    resize(data_dict['pred_target_img']))

                if self.args.use_hq_disc:
                    data_dict['fake_score_gen_mix'], _ = self.discriminator(
                        resize(data_dict['pred_mixing_img']))
                    del _


                loss, losses_dict = self.calc_train_losses(data_dict=data_dict, mode='gen', epoch=epoch,
                                                           ffhq_per_b=ffhq_per_b)
                if self.use_stylegan_d:
                    self.stylegan_discriminator.eval()
                    for p in self.stylegan_discriminator.parameters():
                        p.requires_grad = False

                    data_dict['fake_style_score_gen'] = self.stylegan_discriminator(
                        (data_dict['pred_target_img'] - 0.5) * 2)

                    # print('Mean of discr fake output', torch.mean(data_dict['fake_style_score_gen']))
                    losses_dict["g_style"] = self.weights['stylegan_weight'] * g_nonsaturating_loss(
                        data_dict['fake_style_score_gen'])

                    if self.weights['cycle_idn'] or self.weights['cycle_exp'] and epoch >= self.args.mix_losses_start:
                        data_dict['fake_style_score_gen_mix'] = self.stylegan_discriminator(
                            (data_dict['pred_mixing_img'] - 0.5) * 2)

                        losses_dict["g_style"] += self.weights['stylegan_weight'] * g_nonsaturating_loss(
                            data_dict['fake_style_score_gen_mix'])
                    # losses_dict["g_style"] = self.weights['stylegan_weight'] * self.adversarial_loss([[data_dict['fake_style_score_gen']]], mode='gen')



            elif phase == 'test':
                loss = None
                losses_dict = self.calc_test_losses(data_dict)

        elif mode == 'dis':

            # Backward through dis
            self.discriminator.train()
            for p in self.discriminator.parameters():
                p.requires_grad = True

            # if self.args.use_hq_disc:
            #     self.discriminator2.train()
            #     for p in self.discriminator2.parameters():
            #         p.requires_grad = True
            # person_mask, _, _, _ = self.face_idt.forward(data_dict['target_img'])
            # data_dict['target_img_to_dis'] = data_dict['target_img']
            # data_dict['source_img_to_dis'] = data_dict['pred_target_img'].detach()
            data_dict['real_score_dis'], _ = self.discriminator(resize(data_dict['target_img']))
            # data_dict['real_score_dis'], _ = self.discriminator(data_dict['target_img'])
            data_dict['fake_score_dis'], _ = self.discriminator(resize(data_dict['pred_target_img'].detach()))


            if self.args.use_hq_disc:
                data_dict['real_score_dis_mix'], _ = self.discriminator(resize(data_dict['target_img']))
                data_dict['fake_score_dis_mix'], _ = self.discriminator(resize(data_dict['pred_mixing_img'].detach()))

            loss, losses_dict = self.calc_train_losses(data_dict=data_dict, mode='dis', ffhq_per_b=ffhq_per_b, epoch=epoch)

        elif mode == 'dis_stylegan':
            losses_dict = {}
            self.stylegan_discriminator.train()
            for p in self.stylegan_discriminator.parameters():
                p.requires_grad = True

            d_regularize = iteration % self.args.d_reg_every == 0

            if d_regularize:
                # print('d_regularize')
                data_dict['target_img'].requires_grad_()
                data_dict['target_img'].retain_grad()

            fake_pred = self.stylegan_discriminator((data_dict['pred_target_img'].detach() - 0.5) * 2)
            real_pred = self.stylegan_discriminator((data_dict['target_img'] - 0.5) * 2)
            # print('Mean of discr fake2 output', torch.mean(fake_pred))
            # print('Mean of discr real output', torch.mean(fake_pred))
            losses_dict["d_style"] = d_logistic_loss(real_pred, fake_pred)

            if self.weights['cycle_idn'] or self.weights['cycle_exp'] and epoch >= self.args.mix_losses_start:
                fake_pred_mix = self.stylegan_discriminator((data_dict['pred_mixing_img'].detach() - 0.5) * 2)
                fake_loss_mix = F.softplus(fake_pred_mix)
                losses_dict["d_style"] += fake_loss_mix.mean()
                # losses_dict["d_style"] = self.weights['stylegan_weight'] * self.adversarial_loss([[real_pred]], [[fake_pred]], mode='dis')

            # if rank==0:
            #     print(f"real_score_style {real_pred.mean()}")
            #     print(f"real_score_style {fake_pred.mean()}")

            # self.stylegan_discriminator.zero_grad()
            # d_loss.backward()
            # optim.step()

            # if apply_r1_penalty:
            #     target_img.requires_grad_()
            #     target_img.retain_grad()

            if d_regularize:
                r1_penalty = _calc_r1_penalty(data_dict['target_img'],
                                              real_pred,
                                              scale_number='all',
                                              )
                data_dict['target_img'].requires_grad_(False)
                losses_dict["r1"] = r1_penalty * self.args.d_reg_every * self.args.r1

            # if d_regularize:
            #     # data_dict['target_img'].requires_grad = True
            #     #
            #     # real_img_aug = data_dict['target_img']
            #     # real_pred = self.stylegan_discriminator(real_img_aug)
            #     r1_loss = d_r1_loss(real_pred, data_dict['target_img'])
            #
            #     # self.stylegan_discriminator.zero_grad()
            #     # (self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]).backward()
            #     losses_dict['r1_reg'] = self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]

            # optim.step()
            # losses_dict["r1"] = r1_loss
            # self.r1_loss = losses_dict['r1_reg'].detach()
            # else:
            #     losses_dict["r1"] = self.r1_loss.to(data_dict['target_img'].device)

            # requires_grad(self.stylegan_discriminator, False)

            loss = 0
            for k, v in losses_dict.items():
                try:
                    loss += v
                except Exception as e:
                    print(e, ' Loss adding error')
                    print(k, v, loss)
                    losses_dict[k] = v[0]
                finally:
                    pass

        visuals = None
        if visualize:
            data_dict = self.visualize_data(data_dict)
            visuals = self.get_visuals(data_dict)
        return loss, losses_dict, visuals, data_dict

    def keypoints_to_heatmaps(self, keypoints, img):
        HEATMAPS_VAR = 1e-2
        s = img.shape[2]

        keypoints = keypoints[..., :2]  # use 2D projection of keypoints

        return self.kp2gaussian(keypoints, img.shape[2:], HEATMAPS_VAR)

    def kp2gaussian(self, kp, spatial_size, kp_variance):
        """
        Transform a keypoint into gaussian like representation
        """
        mean = kp

        coordinate_grid = self.make_coordinate_grid(spatial_size, mean.type())
        number_of_leading_dimensions = len(mean.shape) - 1
        shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
        coordinate_grid = coordinate_grid.view(*shape)
        repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
        coordinate_grid = coordinate_grid.repeat(*repeats)

        # Preprocess kp shape
        shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
        mean = mean.view(*shape)

        mean_sub = (coordinate_grid - mean)

        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

        return out

    def make_coordinate_grid(self, spatial_size, type):
        """
        Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
        """
        h, w = spatial_size
        x = torch.arange(w).type(type)
        y = torch.arange(h).type(type)

        x = (2 * (x / (w - 1)) - 1)
        y = (2 * (y / (h - 1)) - 1)

        yy = y.view(-1, 1).repeat(1, w)
        xx = x.view(1, -1).repeat(h, 1)

        meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

        return meshed

    def visualize_data(self, data_dict):
        return visualize_data(self, data_dict)

    def get_visuals(self, data_dict):
        return get_visuals(self, data_dict)

    @staticmethod
    def draw_stickman(args, poses):
        return draw_stickman(args, poses)

    def gen_parameters(self):

        params = itertools.chain(
            self.warp_embed_head_orig.parameters(),
            # self.warp_idt_s.parameters(),
            # self.warp_idt_d.parameters(),
            # self.pose_embed_decode.parameters(),
            # self.pose_embed_code.parameters(),
            self.idt_embedder.parameters(),
            self.expression_embedder.parameters(),
            self.pose_unsqueeze.parameters(),
            self.uv_generator.parameters(),
            self.xy_generator.parameters(),
            self.volume_process_net.parameters(),
            self.decoder.parameters(),
            self.local_encoder.parameters(),
            # self.local_encoder_mask.parameters(),
            self.volume_renderer.parameters(),

        )

        if self.args.use_tensor:
            params = itertools.chain(
                self.warp_embed_head_orig.parameters(),
                # self.pose_embed_decode.parameters(),
                # self.pose_embed_code.parameters(),
                # self.warp_idt_s.parameters(),
                # self.warp_idt_d.parameters(),
                self.idt_embedder.parameters(),
                self.expression_embedder.parameters(),
                self.pose_unsqueeze.parameters(),
                self.uv_generator.parameters(),
                self.xy_generator.parameters(),
                self.volume_process_net.parameters(),
                self.volume_source_net.parameters(),
                self.decoder.parameters(),
                self.local_encoder.parameters(),
                self.volume_renderer.parameters(),
                self.avarage_tensor
                # self.local_encoder_mask.parameters(),
            )





        if self.args.source_volume_num_blocks > 0:
            params = itertools.chain(
                self.warp_embed_head_orig.parameters(),
                # self.pose_embed_decode.parameters(),
                # self.pose_embed_code.parameters(),
                # self.warp_idt_s.parameters(),
                # self.warp_idt_d.parameters(),
                self.idt_embedder.parameters(),
                self.expression_embedder.parameters(),
                self.pose_unsqueeze.parameters(),
                self.uv_generator.parameters(),
                self.xy_generator.parameters(),
                self.volume_process_net.parameters(),
                self.volume_source_net.parameters(),
                self.decoder.parameters(),
                self.local_encoder.parameters(),
                self.volume_renderer.parameters(),
                # self.local_encoder_mask.parameters(),
            )
            if self.args.pred_volume_num_blocks > 0:
                params = itertools.chain(
                    self.warp_embed_head_orig.parameters(),
                    # self.pose_embed_decode.parameters(),
                    # self.pose_embed_code.parameters(),
                    # self.warp_idt_s.parameters(),
                    # self.warp_idt_d.parameters(),
                    self.idt_embedder.parameters(),
                    self.expression_embedder.parameters(),
                    self.pose_unsqueeze.parameters(),
                    self.uv_generator.parameters(),
                    self.xy_generator.parameters(),
                    self.volume_process_net.parameters(),
                    self.volume_source_net.parameters(),
                    self.volume_pred_net.parameters(),
                    self.decoder.parameters(),
                    self.local_encoder.parameters(),
                    self.volume_renderer.parameters(),
                    # self.local_encoder_mask.parameters(),
                )


            if self.args.less_em:
                params = itertools.chain(
                    self.warp_embed_head_orig.parameters(),
                    self.pose_embed_decode.parameters(),
                    self.pose_embed_code.parameters(),
                    # self.warp_idt_s.parameters(),
                    # self.warp_idt_d.parameters(),
                    self.idt_embedder.parameters(),
                    self.expression_embedder.parameters(),
                    self.pose_unsqueeze.parameters(),
                    self.uv_generator.parameters(),
                    self.xy_generator.parameters(),
                    self.volume_process_net.parameters(),
                    self.volume_source_net.parameters(),
                    self.decoder.parameters(),
                    self.local_encoder.parameters(),
                    self.volume_renderer.parameters(),
                    # self.local_encoder_mask.parameters(),
                )
        if self.use_seg and self.args.use_back:
            params = itertools.chain(
                self.warp_embed_head_orig.parameters(),
                self.idt_embedder.parameters(),
                self.expression_embedder.parameters(),
                self.pose_unsqueeze.parameters(),
                self.uv_generator.parameters(),
                self.xy_generator.parameters(),
                self.volume_process_net.parameters(),
                self.decoder.parameters(),
                self.backgroung_adding.parameters(),
                self.background_process_net.parameters(),
                self.local_encoder.parameters(),
                self.local_encoder_seg.parameters(),
                self.volume_renderer.parameters(),
                # self.local_encoder_mask.parameters()
            )

        for param in params:
            yield param

    def configure_optimizers(self):
        self.optimizer_idx_to_mode = {0: 'gen', 1: 'dis', 2: 'dis_stylegan'}

        opts = {
            'adam': lambda param_groups, lr, beta1, beta2: torch.optim.Adam(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2),
                # eps=1e-8
            ),
            'adamw': lambda param_groups, lr, beta1, beta2: torch.optim.AdamW(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2))}

        opt_gen = opts[self.args.gen_opt_type](
            self.gen_parameters(),
            self.args.gen_lr,
            self.args.gen_beta1,
            self.args.gen_beta2)

        if self.args.use_hq_disc:
            opt_dis = opts[self.args.dis_opt_type](
                # itertools.chain(self.discriminator.parameters(), self.discriminator2.parameters()),
                itertools.chain(self.discriminator.parameters(),),
                self.args.dis_lr,
                self.args.dis_beta1,
                self.args.dis_beta2,
            )
        else:
            opt_dis = opts[self.args.dis_opt_type](
                self.discriminator.parameters(),
                self.args.dis_lr,
                self.args.dis_beta1,
                self.args.dis_beta2,
            )

        if self.use_stylegan_d:
            d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)
            opt_dis_style = opts['adam'](
                self.stylegan_discriminator.parameters(),
                self.args.dis_stylegan_lr * d_reg_ratio,
                0 ** d_reg_ratio,
                0.99 ** d_reg_ratio,
            )
            return [opt_gen, opt_dis, opt_dis_style]
        else:
            return [opt_gen, opt_dis]

    def configure_schedulers(self, opts, epochs=None, steps_per_epoch=None):
        shds = {
            'step': lambda optimizer, lr_max, lr_min, max_iters, epochs,
                           steps_per_epoch: torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=max_iters,
                gamma=lr_max / lr_min),

            'cosine': lambda optimizer, lr_max, lr_min, max_iters, epochs,
                             steps_per_epoch: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max_iters,
                eta_min=lr_min),
            'onecycle': lambda optimizer, lr_max, lr_min, max_iters, epochs,
                               steps_per_epoch: torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=lr_max,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1
            )}

        shd_gen = shds[self.args.gen_shd_type](
            opts[0],
            self.args.gen_lr,
            self.args.gen_shd_lr_min,
            self.args.gen_shd_max_iters,
            epochs,
            steps_per_epoch)

        shd_dis = shds[self.args.dis_shd_type](
            opts[1],
            self.args.dis_lr,
            self.args.dis_shd_lr_min,
            self.args.dis_shd_max_iters,
            epochs,
            steps_per_epoch
        )

        if self.use_stylegan_d:
            shd_dis_stylegan = shds[self.args.dis_shd_type](
                opts[2],
                self.args.dis_stylegan_lr,
                self.args.dis_shd_lr_min,
                self.args.dis_shd_max_iters,
                epochs,
                steps_per_epoch
            )

            return [shd_gen, shd_dis, shd_dis_stylegan], [self.args.gen_shd_max_iters, self.args.dis_shd_max_iters,
                                                          self.args.dis_shd_max_iters]
        else:
            return [shd_gen, shd_dis], [self.args.gen_shd_max_iters, self.args.dis_shd_max_iters]
