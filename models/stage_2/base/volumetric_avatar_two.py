import copy

import torch
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
import albumentations as A
import numpy as np
import itertools
from torch.cuda import amp
from networks import basic_avatar, volumetric_avatar
from utils import args as args_utils
from utils import spectral_norm, weight_init, point_transforms
from skimage.measure import label
from .va_losses_and_visuals_two import calc_train_losses, calc_test_losses, prepare_input_data, MODNET, init_losses
from .va_losses_and_visuals_two import visualize_data, get_visuals, draw_stickman
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
        parser.add_argument('--head_pose_regressor_path', default='/fsx/nikitadrobyshev/EmoPortraits/head_pose_regressor.pth')
        parser.add_argument('--additive_motion', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--use_seg', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_back', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_mix_mask', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_masked_aug', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--resize_depth', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--volume_renderer_mode', default='depth_to_channels', type=str)

        # parser.add_argument('--decoder_num_bottleneck_groups', default=6, type=int)

        parser.add_argument('--init_type', default='kaiming')
        parser.add_argument('--init_gain', default=0.0, type=float)

        parser.add_argument('--resize_s2', default=128, type=int)
        parser.add_argument('--dis_num_channels', default=64, type=int)
        parser.add_argument('--dis_max_channels', default=512, type=int)
        parser.add_argument('--dec_max_channels', default=512, type=int)
        parser.add_argument('--dec_max_channels2', default=512, type=int)
        parser.add_argument('--dis2_num_channels', default=64, type=int)
        parser.add_argument('--dis2_max_channels', default=512, type=int)
        # parser.add_argument('--dis_num_blocks', default=4, nargs="+", type=int)
        parser.add_argument('--dis_num_blocks', default=4, type=int)
        parser.add_argument('--dis2_num_blocks', default=4, type=int)

        parser.add_argument('--dis_num_blocks_s2', default=4, type=int)
        parser.add_argument('--dis2_num_blocks_s2', default=4, type=int)

        parser.add_argument('--dis_num_scales', default=1, type=int)
        parser.add_argument('--dis2_num_scales', default=1, type=int)

        parser.add_argument('--use_hq_disc', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_second_dis', default='True', type=args_utils.str2bool, choices=[True, False])




        parser.add_argument('--dis_init_type', default='xavier')
        parser.add_argument('--dis_init_gain', default=0.02, type=float)

        parser.add_argument('--adversarial_weight', default=1.0, type=float)
        parser.add_argument('--adversarial_gen', default=1.0, type=float)
        parser.add_argument('--adversarial_gen_2', default=1.0, type=float)
        parser.add_argument('--feature_matching_weight', default=60.0, type=float)
        parser.add_argument('--vgg19_weight', default=20.0, type=float)
        parser.add_argument('--l1_weight', default=0.0, type=float)


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


        parser.add_argument('--l1_back', default=0.0, type=float)
        parser.add_argument('--cycle_idn', default=0.0, type=float)
        parser.add_argument('--cycle_exp', default=0.0, type=float)
        parser.add_argument('--vgg19_weight_cycle_idn', default=0.0, type=float)
        parser.add_argument('--vgg19_face_cycle_idn', default=0.0, type=float)
        parser.add_argument('--vgg19_weight_cycle_exp', default=0.0, type=float)
        parser.add_argument('--vgg19_face_cycle_exp', default=0.0, type=float)
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
                            default='local_encoder, local_encoder_seg, local_encoder_mask, idt_embedder, expression_embedder, xy_generator, uv_generator, warp_embed_head_orig, pose_embed_decode, pose_embed_code, volume_process_net, volume_source_net, decoder, backgroung_adding, background_process_net')
        parser.add_argument('--ws_networks',
                            default='local_encoder, local_encoder_seg, local_encoder_mask, idt_embedder, expression_embedder, xy_generator, uv_generator, warp_embed_head_orig,  pose_embed_decode, pose_embed_code, volume_process_net, volume_source_net, decoder, backgroung_adding, background_process_net')
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


        parser.add_argument('--norm_layer_dis', default='in')
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
        parser.add_argument('--cycle_stage2', default=0, type=float)

        # Gen parametres
        parser.add_argument('--gen_num_channels', default=32, type=int)
        parser.add_argument('--gen_max_channels', default=512, type=int)
        parser.add_argument('--gen_activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--gen_downsampling_type', default='avgpool', type=str)
        parser.add_argument('--gen_upsampling_type', default='trilinear', type=str)
        parser.add_argument('--gen_pred_flip', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--gen_pred_mixing', default='True', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--pred_mixing_stage2', default='True', type=args_utils.str2bool, choices=[True, False])

        # parser.add_argument('--gen_input_image_size', default=256, type=int)
        parser.add_argument('--idt_image_size', default=256, type=int)
        parser.add_argument('--exp_image_size', default=256, type=int)
        parser.add_argument('--image_additional_size', default=None, type=int)




        parser.add_argument('--dec_num_blocks_stage2', default=8, type=int)
        parser.add_argument('--dec_channel_mult_stage2', default=4.0, type=float)
        parser.add_argument('--enc_channel_mult_stage2', default=4.0, type=float)
        parser.add_argument('--diff_ratio', default=5.0, type=float)
        parser.add_argument('--mask_threshold', default=0.01, type=float)
        parser.add_argument('--enc_init_num_channels_stage2', default=512, type=int)

        parser.add_argument('--source_volume_num_blocks_first', default=2, type=int)
        parser.add_argument('--source_volume_num_blocks_second', default=8, type=int)
        parser.add_argument('--input_size', default=256, type=int)
        parser.add_argument('--output_size', default=512, type=int)
        parser.add_argument('--output_size_s2', default=512, type=int)

        parser.add_argument('--gen_latent_texture_size', default=64, type=int)
        parser.add_argument('--gen_latent_texture_size2', default=64, type=int)
        parser.add_argument('--gen_latent_texture_depth', default=16, type=int)
        parser.add_argument('--gen_latent_texture_channels', default=64, type=int)
        parser.add_argument('--gen_latent_texture_channels2', default=64, type=int)

        parser.add_argument('--latent_volume_channels', default=64, type=int)
        parser.add_argument('--latent_volume_size', default=64, type=int)
        parser.add_argument('--latent_volume_depth', default=16, type=int)
        parser.add_argument('--source_volume_num_blocks', default=0, type=int)


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
        parser.add_argument('--sep_train_losses', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dec_conf_ms_names', default='target_vgg19_conf, target_fem_conf', type=str)
        parser.add_argument('--dec_conf_names', default='', type=str)
        parser.add_argument('--dec_conf_ms_scales', default=5, type=int)
        parser.add_argument('--dec_conf_channel_mult', default=1.0, type=float)

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
        parser.add_argument('--min_log_stylegan', default=2, type=int)

        parser.add_argument('--lpe_head_transform_sep_scales', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--num_b_negs', default=1, type=int)

        parser.add_argument('--use_stylegan_d', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_stylegan_d_stage2', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dis_stylegan_lr', default=2e-4, type=float)
        parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
        parser.add_argument('--stylegan_weight', default=1.0, type=float)
        parser.add_argument("--r1", type=float, default=0.0, help="weight of the r1 regularization")
        parser.add_argument("--r1_s2", type=float, default=10.0, help="weight of the r1 regularization")

        return parser_out

    def __init__(self, args, training=True, rank=0):
        super(Model, self).__init__()
        self.args = args
        self.init_rank = rank
        self.num_source_frames = args.num_source_frames
        self.num_target_frames = args.num_target_frames
        self.embed_size = args.gen_embed_size
        self.num_source_frames = args.num_source_frames  # number of identities per batch
        self.embed_size = args.gen_embed_size
        self.pred_seg = args.dec_pred_seg
        self.use_stylegan_d = args.use_stylegan_d_stage2
        if self.pred_seg:
            self.seg_loss = nn.BCELoss()
        self.pred_flip = args.gen_pred_flip
        self.pred_mixing = args.gen_pred_mixing
        assert self.num_source_frames == 1, 'No support for multiple sources'
        self.background_net_input_channels = 64

        # self.pred_mixing = args.gen_pred_mixing
        self.weights = {
            'adversarial': args.adversarial_weight,
            'adversarial_gen': args.adversarial_gen,
            'adversarial_gen_2': args.adversarial_gen_2,
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

            'vgg19_face_cycle_idn': args.vgg19_face_cycle_idn,
            'vgg19_cycle_idn': args.vgg19_weight_cycle_idn,

            'vgg19_face_cycle_exp': args.vgg19_face_cycle_exp,
            'vgg19_cycle_exp': args.vgg19_weight_cycle_exp,

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
        self.local_encoder = volumetric_avatar.LocalEncoderOld(
            use_amp_autocast=args.use_amp_autocast,
            gen_upsampling_type=args.gen_upsampling_type,
            gen_downsampling_type=args.gen_downsampling_type,
            gen_input_image_size=args.output_size_s2,
            gen_latent_texture_size=args.gen_latent_texture_size2,
            gen_latent_texture_depth=args.gen_latent_texture_depth,
            warp_norm_grad=args.warp_norm_grad,
            gen_num_channels=args.gen_num_channels,
            enc_channel_mult=args.enc_channel_mult_stage2,
            norm_layer_type=args.norm_layer_type,
            num_gpus=args.num_gpus,
            gen_max_channels=args.gen_max_channels,
            enc_block_type=args.enc_block_type,
            gen_activation_type=args.gen_activation_type,
            gen_latent_texture_channels=args.gen_latent_texture_channels2,
            in_channels=3
        )


        # # self.volume_process_net = volumetric_avatar.Unet3D(
        # #     eps=args.eps,
        # #     num_gpus=args.num_gpus,
        # #     gen_embed_size=args.gen_embed_size,
        # #     gen_adaptive_kernel=args.gen_adaptive_kernel,
        # #     use_amp_autocast=args.use_amp_autocast,
        # #     gen_use_adanorm=args.gen_use_adanorm,
        # #     gen_use_adaconv=args.gen_use_adaconv,
        # #     gen_upsampling_type=args.gen_upsampling_type,
        # #     gen_downsampling_type=args.gen_downsampling_type,
        # #     gen_dummy_input_size=args.gen_dummy_input_size,
        # #     gen_latent_texture_size=args.gen_latent_texture_size,
        # #     gen_latent_texture_depth=args.gen_latent_texture_depth,
        # #     gen_adaptive_conv_type=args.gen_adaptive_conv_type,
        # #     gen_latent_texture_channels=args.gen_latent_texture_channels,
        # #     gen_activation_type=args.gen_activation_type,
        # #     gen_max_channels=args.gen_max_channels,
        # #     warp_norm_grad=args.warp_norm_grad,
        # #     warp_block_type=args.warp_block_type,
        # #     tex_pred_rgb=args.tex_pred_rgb,
        # #     image_size=args.image_size,
        # #     tex_use_skip_resblock=args.tex_use_skip_resblock,
        # #     norm_layer_type=args.norm_layer_type,
        # # )
        #

        #
        #
        # if self.args.source_volume_num_blocks_first>0:
        #     self.volume_source_net = volumetric_avatar.ResBlocks3d(
        #         num_gpus=args.num_gpus,
        #         norm_layer_type=args.norm_layer_type,
        #         input_channels=self.args.gen_latent_texture_channels2,
        #         num_blocks=self.args.source_volume_num_blocks_first,
        #         activation_type=args.gen_activation_type,
        #         conv_layer_type='conv_3d',
        #         channels=None,
        #         # channels=[self.args.latent_volume_channels, 2*self.args.latent_volume_channels, 4*self.args.latent_volume_channels, 4*self.args.latent_volume_channels, 2*self.args.latent_volume_channels, self.args.latent_volume_channels],
        #         num_layers=3,
        #
        #     )
        #
        # self.volume_process_net = volumetric_avatar.ResBlocks3d(
        #         num_gpus=args.num_gpus,
        #         norm_layer_type=args.norm_layer_type,
        #         input_channels=self.args.gen_latent_texture_channels2,
        #         num_blocks=self.args.source_volume_num_blocks_second,
        #         activation_type=args.gen_activation_type,
        #         conv_layer_type='conv_3d',
        #         # channels=None,
        #         channels=[64, 64, 96, 64, 64],
        #         # channels=[self.args.latent_volume_channels, 2*self.args.latent_volume_channels, 4*self.args.latent_volume_channels, 4*self.args.latent_volume_channels, 2*self.args.latent_volume_channels, self.args.latent_volume_channels],
        #         num_layers=3,)



        self.decoder = volumetric_avatar.Decoder_stage2Old(
            eps=args.eps,
            image_size=args.output_size_s2,
            use_amp_autocast=args.use_amp_autocast,
            gen_embed_size=args.gen_embed_size,
            gen_adaptive_kernel=args.gen_adaptive_kernel,
            gen_adaptive_conv_type=args.gen_adaptive_conv_type,
            gen_latent_texture_size=args.gen_latent_texture_size2,
            in_channels=args.gen_latent_texture_channels2 * args.gen_latent_texture_depth,
            gen_num_channels=args.gen_num_channels,
            # dec_max_channels=args.gen_max_channels,
            dec_max_channels=args.dec_max_channels2,
            gen_use_adanorm=False,
            gen_activation_type=args.gen_activation_type,
            gen_use_adaconv=args.gen_use_adaconv,
            dec_channel_mult=args.dec_channel_mult_stage2,
            dec_num_blocks=args.dec_num_blocks_stage2,
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
            norm_layer_type=args.norm_layer_type)


        if args.warp_norm_grad:
            self.grid_sample = volumetric_avatar.GridSample(args.gen_latent_texture_size)
        else:
            self.grid_sample = lambda inputs, grid: F.grid_sample(inputs.float(), grid.float(),
                                                                  padding_mode='reflection')

        self.get_mask = MODNET()


        self.num_b_negs = self.args.num_b_negs


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
                         'xy_generator', 'uv_generator', 'warp_embed_head_orig', 'volume_process_net', 'volume_source_net',
                         'decoder', 'backgroung_adding', 'background_process_net']:
            try:
                net = getattr(self, net_name)
                pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                if self.init_rank==0:
                    print(f'Number of trainable params in {net_name} stage 2 : {pytorch_total_params}')
            except Exception as e:
                if self.init_rank==0:
                    print(e)

        if training:
            if self.args.adversarial_weight>0:
                self.discriminator = basic_avatar.MultiScaleDiscriminator(
                    min_channels=args.dis_num_channels,
                    max_channels=args.dis_max_channels,
                    num_blocks=args.dis_num_blocks_s2,
                    input_channels=3,
                    input_size=args.output_size_s2,
                    num_scales=args.dis2_num_scales,
                    norm_layer=self.args.norm_layer_dis)

                self.discriminator.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))

            if self.args.use_second_dis:
                self.discriminator2 = basic_avatar.MultiScaleDiscriminator(
                    min_channels=args.dis2_num_channels,
                    max_channels=args.dis2_max_channels,
                    num_blocks=args.dis2_num_blocks_s2,
                    input_channels=3,
                    input_size=args.output_size_s2,
                    num_scales=args.dis2_num_scales,
                    norm_layer=self.args.norm_layer_dis)

                self.discriminator2.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))
                pytorch_total_params = sum(p.numel() for p in self.discriminator2.parameters() if p.requires_grad)
                if self.init_rank == 0:
                    print(f'Number of trainable params in discriminator2 : {pytorch_total_params}')

            if self.args.adversarial_weight > 0:
                pytorch_total_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
                if self.init_rank==0:
                    print(f'Number of trainable params in discriminator : {pytorch_total_params}')

            if self.use_stylegan_d:
                self.r1_loss = torch.tensor(0.0)
                self.stylegan_discriminator = basic_avatar.DiscriminatorStyleGAN2(size=self.args.output_size_s2,
                                                                                  channel_multiplier=2, my_ch=2, min_log=self.args.min_log_stylegan)
                # self.stylegan_discriminator.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))
                pytorch_total_params = sum(
                    p.numel() for p in self.stylegan_discriminator.parameters() if p.requires_grad)
                print(f'Number of trainable params in stylegan2 discriminator : {pytorch_total_params}')

        # if self.separate_idt:
        self.face_idt = volumetric_avatar.FaceParsing(None, 'cuda')

        # self.get_face_vector = volumetric_avatar.utils.Face_vector(self.head_pose_regressor, half=False)
        # self.get_face_vector_resnet = volumetric_avatar.utils.Face_vector_resnet(half=False)

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
                    pass
                    # print(e)
                    # print(f'there is no {net_name} network')

        if args.use_ws:
            ws_nets_names = args_utils.parse_str_to_list(args.ws_networks, sep=',')
            for net_name in ws_nets_names:
                try:
                    net = getattr(self, net_name)
                    # print('apply ws to: ', net_name)
                    new_net = volumetric_avatar.utils.replace_conv_to_ws_conv(net, conv2d=True, conv3d=True)
                    setattr(self, net_name, new_net)
                    # net.apply(lambda module: volumetric_avatar.utils.replace_conv_to_ws_conv(module))
                except Exception as e:
                    pass
                    # print(e)
                    # print(f'there is no {net_name} network')

    def init_losses(self, args):
        return init_losses(self, args)

    @torch.no_grad()
    def get_face_warp(self, grid, params_ffhq):
        grid = grid.view(grid.shape[0], -1, 2)
        face_warp = point_transforms.align_ffhq_with_zoom(grid, params_ffhq)
        face_warp = face_warp.view(face_warp.shape[0], self.args.aug_warp_size, self.args.aug_warp_size, 2)

        return face_warp

    def _forward(self, data_dict, visualize, ffhq_per_b=0, iteration = 0, pred =True, mix = True):
        self.visualize = visualize
        b = data_dict['source_img_ffhq'].shape[0]

        self.pred = pred
        self.mix = mix
        # self.mix = pred
        resize_input = lambda img: F.interpolate(img, mode='bilinear', size=(self.args.output_size_s2//2, self.args.output_size_s2//2), align_corners=False)
        resize = lambda img: F.interpolate(img, mode='bilinear', size=(self.args.output_size_s2, self.args.output_size_s2), align_corners=False)
        resize_in = lambda img: F.interpolate(img, mode='bilinear', size=(self.args.input_size, self.args.input_size))
        resize_n = lambda img: F.interpolate(img, mode='nearest', size=(self.args.output_size_s2, self.args.output_size_s2))

        c = self.args.gen_latent_texture_channels2
        s = self.args.gen_latent_texture_size2
        d = self.args.latent_volume_depth

        resize_d = lambda img: F.interpolate(img, mode='bilinear', size=(self.args.output_size_s2//4, self.args.output_size_s2//4), align_corners=False)
        resize_up = lambda img: F.interpolate(img, mode='bilinear', size=(self.args.output_size_s2, self.args.output_size_s2), align_corners=False)

        resize_v = lambda img: F.interpolate(img, mode='bilinear', size=(s, s), align_corners=False)

        data_dict['resized_pred_target_img'] = resize(data_dict['pred_target_img'])
        data_dict['resized_pred_target_mask'] = self.get_mask.forward(data_dict['resized_pred_target_img'])
        face_mask_source, _, _, _ = self.face_idt.forward(data_dict['resized_pred_target_img'])
        data_dict['resized_pred_target_face_mask'] = data_dict['resized_pred_target_mask'] * face_mask_source
        if self.pred:
            # source_latent_volume = self.local_encoder(data_dict['source_img_ffhq'] * data_dict['source_mask_ffhq'])
            # source_latent_volume = self.local_encoder(torch.cat([data_dict['source_img_ffhq'] * data_dict['source_mask_ffhq'],
            #                               data_dict['resized_pred_target_img'] * data_dict['resized_pred_target_mask']],
            #                              dim=1))

            aligned_target_volume = self.local_encoder(data_dict['resized_pred_target_img'] * data_dict['resized_pred_target_mask'])
            # source_latent_volume = self.local_encoder(torch.cat([data_dict['source_img_ffhq'] * data_dict['source_mask_ffhq'], data_dict['resized_pred_target_img']*data_dict['resized_pred_target_mask']], dim=1))

            # print(source_latent_volume.shape, data_dict['deep_f'].shape, data_dict['img_f'].shape)
            # Reshape latents into 3D volume

            # print(aligned_target_volume.shape)

            # source_latent_volume = source_latent_volume.view(b, c, d, s, s)
            #
            # if self.args.gen_latent_texture_size2>64:
            #     # print('aaa', data_dict['source_rotation_warp'].shape)
            #     resize_w = lambda img: F.interpolate(img, mode='trilinear', size=(self.args.latent_volume_depth, self.args.gen_latent_texture_size2, self.args.gen_latent_texture_size2))
            #     move_a = lambda warp: torch.moveaxis(resize_w(torch.moveaxis(warp, 4, 1)), 1, 4)
            #
            #     a_list = ['source_rotation_warp', 'source_xy_warp_resize', 'target_uv_warp_resize', 'target_rotation_warp', 'mixing_uv_warp_resize', 'mixing_align_warp_resize']
            #
            #     for name in a_list:
            #         if data_dict[name].shape[-2]<self.args.gen_latent_texture_size2:
            #             try:
            #                 data_dict[name] = move_a(data_dict[name])
            #             except Exception as e:
            #                 grid_s = torch.linspace(-1, 1, self.args.gen_latent_texture_size2)
            #                 grid_z = torch.linspace(-1, 1, self.args.latent_volume_depth)
            #                 w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
            #                 one = torch.ones_like(u)
            #                 print(e, data_dict[name].shape)
            #                 data_dict[name] = torch.stack([u, v, w, one], dim=3).view(1, -1, 4)
            #
            #     # print('bbb', data_dict['source_rotation_warp'])
            #
            #     # data_dict['source_xy_warp_resize'] = move_a(data_dict['source_xy_warp_resize'])
            #     # data_dict['target_uv_warp_resize'] = move_a(data_dict['target_uv_warp_resize'])
            #     # data_dict['target_rotation_warp'] = move_a(data_dict['target_rotation_warp'])
            #     # data_dict['mixing_uv_warp_resize'] = move_a(data_dict['mixing_uv_warp_resize'])
            #     # data_dict['mixing_align_warp_resize'] = move_a(data_dict['mixing_align_warp_resize'])
            #
            #
            # if self.args.source_volume_num_blocks_first > 0:
            #     source_latent_volume = self.volume_source_net(source_latent_volume)
            #
            # source_latent_volume = self.grid_sample(
            #     self.grid_sample(source_latent_volume, data_dict['source_rotation_warp']),
            #     data_dict['source_xy_warp_resize'])
            #
            #
            #
            # target_latent_volume = self.volume_process_net(source_latent_volume)
            #
            # aligned_target_volume = self.grid_sample(
            #     self.grid_sample(target_latent_volume, data_dict['target_uv_warp_resize']),
            #     data_dict['target_rotation_warp'])
            #
            # aligned_target_volume = aligned_target_volume.view(b, c * d, s, s)




            # aligned_target_volume = torch.cat([aligned_target_volume, resize_v(data_dict['deep_f'])], dim=1)
            # aligned_target_volume = torch.cat([aligned_target_volume, data_dict['img_f'][0]], dim=1)
            # aligned_target_volume = torch.cat([aligned_target_volume, data_dict['deep_f']], dim=1)
            data_dict['pred_target_add'], _, _, _ = self.decoder(None, None, aligned_target_volume, False, pred_feat=data_dict['img_f'][-1])
            data_dict['pred_target_add'] = data_dict['pred_target_add']*data_dict['resized_pred_target_face_mask']
            data_dict['pred_target_img_ffhq'] = data_dict['resized_pred_target_img'] + data_dict['pred_target_add']
            data_dict['pred_target_img_ffhq'].clamp_(max=1, min=0)

        data_dict['target_img_ffhq'] = data_dict['target_img_ffhq'] * data_dict['target_mask_ffhq'].detach()
        data_dict['resized_target_img_ffhq'] = resize(resize_input(data_dict['target_img_ffhq']))
        data_dict['target_add_ffhq'] = data_dict['target_img_ffhq'] - data_dict['resized_target_img_ffhq']
        data_dict['target_add_ffhq_pred'] = (data_dict['target_img_ffhq'] - data_dict['resized_pred_target_img'])*data_dict['resized_pred_target_face_mask']
        mt = self.args.mask_threshold
        data_dict['target_add_ffhq_pred_mask'] = (torch.sum(data_dict['target_add_ffhq_pred']>mt, dim=1).unsqueeze(1)>0).float()
        # data_dict['target_add_ffhq_pred_mask'] = torch.sum(data_dict['target_add_ffhq_pred'] > 0.01, dim=1).unsqueeze(1).float() - torch.sum(data_dict['target_add_ffhq_pred'] > 0.15, dim=1).unsqueeze(1).float()
        # data_dict['target_add_ffhq_pred_mask2'] = (torch.sum(resize_up(resize_d(data_dict['target_add_ffhq_pred'])) > 0.05, dim=1).unsqueeze(1)>0).float()
        # data_dict['target_add_ffhq_pred_mask'] = data_dict['target_add_ffhq_pred_mask1']*(1-data_dict['target_add_ffhq_pred_mask2'])
        # print(data_dict['target_add_ffhq_pred_mask'].shape)

        # torch.cat((data_dict['resized_pred_target_img'], data_dict['pred_target_add']), dim=1)
        # torch.cat((data_dict['resized_pred_target_img'], data_dict['target_add_ffhq_pred']), dim=1)
        data_dict['resized_pred_mixing_img'] = resize(data_dict['pred_mixing_img'])
        data_dict['resized_pred_mixing_mask'] = self.get_mask.forward(data_dict['resized_pred_mixing_img'])
        face_mask_source, _, _, _ = self.face_idt.forward(data_dict['resized_pred_mixing_img'])
        data_dict['resized_pred_mixing_face_mask'] = data_dict['resized_pred_mixing_mask'] *face_mask_source
        # Cross-reenactment
        if self.args.pred_mixing_stage2 or not self.training:
            # print((self.weights['cycle_idn'] or self.weights['cycle_exp']) and self.training)
            if self.training and self.mix:
                #######################################################################################################
                # Mixing prediction
                # source_latent_volume = self.local_encoder(torch.cat([data_dict['source_img_ffhq'] * data_dict['source_mask_ffhq'],data_dict['resized_pred_target_img'] * data_dict['resized_pred_target_mask']],dim=1))
                # source_latent_volume = source_latent_volume.view(b, c, d, s, s)
                #
                # if self.args.gen_latent_texture_size2 > 64:
                #     # print('aaa', data_dict['source_rotation_warp'].shape)
                #     resize_w = lambda img: F.interpolate(img, mode='nearest', size=(
                #     self.args.latent_volume_depth, self.args.gen_latent_texture_size2,
                #     self.args.gen_latent_texture_size2))
                #     move_a = lambda warp: torch.moveaxis(resize_w(torch.moveaxis(warp, 4, 1)), 1, 4)
                #
                #     a_list = ['source_rotation_warp', 'source_xy_warp_resize', 'target_uv_warp_resize',
                #               'target_rotation_warp', 'mixing_uv_warp_resize', 'mixing_align_warp_resize']
                #
                #     for name in a_list:
                #         try:
                #             data_dict[name] = move_a(data_dict[name])
                #         except Exception as e:
                #             grid_s = torch.linspace(-1, 1, self.args.gen_latent_texture_size2)
                #             grid_z = torch.linspace(-1, 1, self.args.latent_volume_depth)
                #             w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
                #             one = torch.ones_like(u)
                #             print(e, data_dict[name].shape)
                #             data_dict[name] = torch.stack([u, v, w, one], dim=3).view(1, -1, 4)
                #
                # if self.args.source_volume_num_blocks_first > 0:
                #     source_latent_volume = self.volume_source_net(source_latent_volume)
                #
                # source_latent_volume = self.grid_sample(
                #     self.grid_sample(source_latent_volume, data_dict['source_rotation_warp']),
                #     data_dict['source_xy_warp_resize'])
                #
                # target_latent_volume = self.volume_process_net(source_latent_volume)

                aligned_mixing_feat = self.local_encoder(data_dict['resized_pred_mixing_img'] * data_dict['resized_pred_mixing_mask'])

                # warped_mixing_feat = self.grid_sample(target_latent_volume, data_dict['mixing_uv_warp_resize'])
                # aligned_mixing_feat = self.grid_sample(warped_mixing_feat, data_dict['mixing_align_warp_resize']).view(b, c * d, s,s)
                self.decoder.train()
                # aligned_mixing_feat = torch.cat([aligned_mixing_feat, resize_v(data_dict['deep_f_mix'])], dim=1)
                # aligned_mixing_feat = torch.cat([aligned_mixing_feat, data_dict['img_f_mix'][0]], dim=1)
                data_dict['pred_mixing_add_ffhq'], _, _, _ = self.decoder(data_dict, None, aligned_mixing_feat, False)
                data_dict['pred_mixing_add_ffhq'] = data_dict['pred_mixing_add_ffhq'] * data_dict['resized_pred_mixing_face_mask']
                data_dict['pred_mixing_img_ffhq'] = data_dict['resized_pred_mixing_img'] + data_dict['pred_mixing_add_ffhq']
                data_dict['pred_mixing_img_ffhq'].clamp_(max=1, min=0)

            else:
                with torch.no_grad():


                    aligned_mixing_feat = self.local_encoder(data_dict['resized_pred_mixing_img'] * data_dict['resized_pred_mixing_mask'])
                    # warped_mixing_feat = self.grid_sample(target_latent_volume, data_dict['mixing_uv_warp_resize'])
                    # aligned_mixing_feat = self.grid_sample(warped_mixing_feat, data_dict['mixing_align_warp_resize']).view(b, c * d, s, s)
                    self.decoder.eval()
                    # aligned_mixing_feat = torch.cat([aligned_mixing_feat, resize_v(data_dict['deep_f_mix'])], dim=1)
                    # aligned_mixing_feat = torch.cat([aligned_mixing_feat, data_dict['img_f_mix'][0]], dim=1)
                    data_dict['pred_mixing_add_ffhq'], _, _, _ = self.decoder(data_dict, None, aligned_mixing_feat, False)
                    data_dict['pred_mixing_add_ffhq'] = data_dict['pred_mixing_add_ffhq'] * data_dict['resized_pred_mixing_face_mask']
                    data_dict['pred_mixing_img_ffhq'] = data_dict['resized_pred_mixing_img'] + data_dict['pred_mixing_add_ffhq']
                    data_dict['pred_mixing_img_ffhq'].clamp_(max=1, min=0)



        # data_dict['HF_target_ffhq'] = data_dict['target_img_ffhq'] * data_dict[
        #     'resized_pred_target_face_mask'] - resize_up(resize_d(data_dict['target_img_ffhq'] * data_dict['resized_pred_target_face_mask']))
        # data_dict['HF_pred_ffhq'] = data_dict['pred_target_img_ffhq'] * data_dict[
        #     'resized_pred_target_face_mask'] - resize_up(resize_d(data_dict['pred_target_img_ffhq'] * data_dict['resized_pred_target_face_mask']))
        return data_dict


    def calc_train_losses(self, data_dict: dict, mode: str = 'gen', epoch=0, ffhq_per_b=0, iteration=0):
        return calc_train_losses(self, data_dict=data_dict, mode=mode, epoch=epoch, ffhq_per_b=ffhq_per_b, iteration=iteration)

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
                epoch=0,
                pred=True,
                mix=True):
        assert phase in ['train', 'test']
        mode = self.optimizer_idx_to_mode[optimizer_idx]
        # print(mode)
        self.ffhq_per_b = ffhq_per_b
        s = self.args.image_additional_size if self.args.image_additional_size is not None else data_dict[
            'target_img'].shape[-1]
        resize = lambda img: F.interpolate(img, mode='bilinear', size=(s, s), align_corners=False)

        # if iteration%2==0:
        #     aug_r = A.Compose(
        #         [A.MedianBlur(blur_limit=5, always_apply=True)
        #          ])
        #     device = data_dict['pred_target_img'].device
        #     data_dict['pred_target_img'] = torch.stack([torch.from_numpy(aug_r(image=data_dict['pred_target_img'][i].cpu().detach().numpy())['image']) for i in range(data_dict['pred_target_img'].shape[0])], dim=0).to(device)
        #     data_dict['pred_mixing_img'] = torch.stack([torch.from_numpy(aug_r(image=data_dict['pred_mixing_img'][i].cpu().detach().numpy())['image']) for i in range(data_dict['pred_mixing_img'].shape[0])], dim=0).to(device)

        if mode == 'gen':
            # data_dict = self.prepare_input_data(data_dict)

            if ffhq_per_b == 0:
                data_dict = self._forward(data_dict, visualize, iteration = iteration, pred=pred, mix=mix)
            else:
                data_dict = self._forward(data_dict, visualize, ffhq_per_b=ffhq_per_b, iteration = iteration, pred=pred, mix=mix)

            if phase == 'train':

                if self.args.adversarial_weight > 0:
                    self.discriminator.eval()
                    for p in self.discriminator.parameters():
                        p.requires_grad = False

                if self.args.use_second_dis:
                    self.discriminator2.eval()
                    for p in self.discriminator2.parameters():
                        p.requires_grad = False

                if self.args.adversarial_weight > 0 and self.pred:
                    with torch.no_grad():
                        # person_mask, _, _, _ = self.face_idt.forward(data_dict['target_img'])
                        # _, data_dict['real_feats_gen'] = self.discriminator(torch.cat((data_dict['resized_pred_target_img'], data_dict['target_add_ffhq_pred']), dim=1))
                        _, data_dict['real_feats_gen'] = self.discriminator(data_dict['target_img_ffhq']*data_dict['resized_pred_target_face_mask'])
                        # _, data_dict['real_feats_gen'] = self.discriminator(data_dict['HF_target_ffhq'])


                    # data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(torch.cat((data_dict['resized_pred_target_img'], data_dict['pred_target_add']), dim=1))
                    data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(data_dict['pred_target_img_ffhq']*data_dict['resized_pred_target_face_mask'])
                    # data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(data_dict['pred_mixing_img_ffhq'] * data_dict['resized_pred_mixing_face_mask'])
                    # data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(data_dict['HF_pred_ffhq'])

                if self.args.use_second_dis and self.mix:
                    assert data_dict['pred_mixing_img_ffhq'].requires_grad == True
                    data_dict['fake_score_gen_mix'], _ = self.discriminator2(data_dict['pred_mixing_img_ffhq']*data_dict['resized_pred_mixing_face_mask'])
                    # data_dict['fake_score_gen_mix'], _ = self.discriminator2(data_dict['HF_pred_ffhq'])


                loss, losses_dict = self.calc_train_losses(data_dict=data_dict, mode='gen', epoch=epoch, ffhq_per_b=ffhq_per_b)

                if self.use_stylegan_d and self.pred:
                    self.stylegan_discriminator.eval()
                    for p in self.stylegan_discriminator.parameters():
                        p.requires_grad = False

                    # print(data_dict['pred_mixing_img_ffhq'].requires_grad)
                    # assert data_dict['pred_mixing_img_ffhq'].requires_grad == True
                    assert data_dict['pred_target_img_ffhq'].requires_grad == True
                    # data_dict['fake_style_score_gen'] = self.stylegan_discriminator((data_dict['pred_mixing_img_ffhq']*data_dict['resized_pred_mixing_face_mask'] - 0.5) * 2)
                    data_dict['fake_style_score_gen'] = self.stylegan_discriminator((data_dict['pred_target_img_ffhq']*data_dict['resized_pred_target_face_mask'] - 0.5) * 2)

                    # print('Mean of discr fake output', torch.mean(data_dict['fake_style_score_gen']))
                    losses_dict["g_style"] = self.weights['stylegan_weight'] * g_nonsaturating_loss(data_dict['fake_style_score_gen'])

                    # losses_dict['g_style'] = self.adversarial_loss(
                    #             fake_scores=[[data_dict['fake_style_score_gen'],]],
                    #             mode='gen')

                    # if self.weights['cycle_idn'] or self.weights['cycle_exp'] and epoch >= self.args.mix_losses_start:
                    #     data_dict['fake_style_score_gen_mix'] = self.stylegan_discriminator(
                    #         (data_dict['pred_mixing_img'] - 0.5) * 2)
                    #
                    #     losses_dict["g_style"] += self.weights['stylegan_weight'] * g_nonsaturating_loss(
                    #         data_dict['fake_style_score_gen_mix'])

                    # losses_dict["g_style"] = self.weights['stylegan_weight'] * self.adversarial_loss([[data_dict['fake_style_score_gen']]], mode='gen')

            elif phase == 'test':
                loss = None
                losses_dict = self.calc_test_losses(data_dict)

        elif mode == 'dis':

            # Backward through dis
            if self.args.adversarial_weight > 0:
                self.discriminator.train()
                for p in self.discriminator.parameters():
                    p.requires_grad = True



                # data_dict['real_score_dis'], _ = self.discriminator(torch.cat((data_dict['resized_pred_target_img'], data_dict['target_add_ffhq_pred']), dim=1))
                # data_dict['fake_score_dis'], _ = self.discriminator(torch.cat((data_dict['resized_pred_target_img'], data_dict['pred_target_add']), dim=1).detach())

                data_dict['real_score_dis'], _ = self.discriminator(data_dict['target_img_ffhq']*data_dict['resized_pred_target_face_mask'])
                data_dict['fake_score_dis'], _ = self.discriminator(data_dict['pred_target_img_ffhq'].detach()*data_dict['resized_pred_target_face_mask'].detach())
                # data_dict['fake_score_dis'], _ = self.discriminator(data_dict['pred_mixing_img_ffhq'].detach() * data_dict['resized_pred_mixing_face_mask'].detach())

                # data_dict['real_score_dis'], _ = self.discriminator(data_dict['HF_target_ffhq'])
                # data_dict['fake_score_dis'], _ = self.discriminator(data_dict['HF_pred_ffhq'].detach())

                # for i, s in enumerate(data_dict['fake_score_dis']):
                #     for ii, ss in enumerate(s):
                #         print(ss.shape, f'{i}_{ii}_d')

            if self.args.use_second_dis:
                self.discriminator2.train()
                for p in self.discriminator2.parameters():
                    p.requires_grad = True

            if self.args.use_second_dis:
                # print(iteration)
                data_dict['real_score_dis_mix'], _ = self.discriminator2(data_dict['target_img_ffhq']*data_dict['resized_pred_target_face_mask'])
                data_dict['fake_score_dis_mix'], _ = self.discriminator2(data_dict['pred_mixing_img_ffhq'].detach()*data_dict['resized_pred_mixing_face_mask'].detach())
                # data_dict['fake_score_dis_mix'], _ = self.discriminator(data_dict['pred_target_img_ffhq'].detach() * data_dict['resized_pred_target_face_mask'].detach())
                # data_dict['real_score_dis_mix'], _ = self.discriminator2(data_dict['HF_target_ffhq'])
                # data_dict['fake_score_dis_mix'], _ = self.discriminator2(data_dict['HF_pred_ffhq'].detach())

            loss, losses_dict = self.calc_train_losses(data_dict=data_dict, mode='dis', ffhq_per_b=ffhq_per_b, epoch=epoch, iteration=iteration)


        elif mode == 'dis_stylegan':
            losses_dict = {}
            self.stylegan_discriminator.train()
            for p in self.stylegan_discriminator.parameters():
                p.requires_grad = True

            d_regularize = iteration % self.args.d_reg_every == 0

            if d_regularize:
                # print('d_regularize')
                data_dict['target_img_ffhq'].requires_grad_()
                data_dict['target_img_ffhq'].retain_grad()

            # fake_pred = self.stylegan_discriminator((data_dict['pred_mixing_img_ffhq'].detach()*data_dict['resized_pred_mixing_face_mask'].detach() - 0.5) * 2)
            fake_pred = self.stylegan_discriminator((data_dict['pred_target_img_ffhq'].detach() * data_dict['resized_pred_target_face_mask'].detach() - 0.5) * 2)
            real_pred = self.stylegan_discriminator((data_dict['target_img_ffhq']*data_dict['resized_pred_target_face_mask'] - 0.5) * 2)

            # fake_pred = self.stylegan_discriminator((data_dict['HF_pred_ffhq'].detach() - 0.5) * 2)
            # real_pred = self.stylegan_discriminator((data_dict['HF_target_ffhq'] - 0.5) * 2)
            # print('Mean of discr fake2 output', torch.mean(fake_pred))
            # print('Mean of discr real output', torch.mean(fake_pred))
            losses_dict["d_style"] = d_logistic_loss(real_pred, fake_pred)
            #
            # losses_dict['d_style'] = self.adversarial_loss(
            #             real_scores=[[real_pred,]],
            #             fake_scores=[[fake_pred,]],
            #             mode='dis')

            # if self.weights['cycle_idn'] or self.weights['cycle_exp'] and epoch >= self.args.mix_losses_start:
            #     fake_pred_mix = self.stylegan_discriminator((data_dict['pred_mixing_img'].detach() - 0.5) * 2)
            #     fake_loss_mix = F.softplus(fake_pred_mix)
            #     losses_dict["d_style"] += fake_loss_mix.mean()
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
                r1_penalty = _calc_r1_penalty(data_dict['target_img_ffhq'],
                                              real_pred,
                                              scale_number='all',
                                              )
                data_dict['target_img_ffhq'].requires_grad_(False)
                losses_dict["r1"] = r1_penalty * self.args.d_reg_every * self.args.r1_s2

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
            self.local_encoder.parameters(),
            # self.volume_process_net.parameters(),
            self.decoder.parameters(),

        )

        if self.args.source_volume_num_blocks > 0:
            params = itertools.chain(
                self.local_encoder.parameters(),
                # self.volume_process_net.parameters(),
                # self.volume_source_net.parameters(),
                self.decoder.parameters(),
            )


        for param in params:
            yield param

    def configure_optimizers(self):

        # self.optimizer_idx_to_mode = {0: 'gen',  1: 'dis_stylegan'}
        # self.optimizer_idx_to_mode = {0: 'gen', 1: 'dis', 2: 'dis_stylegan'}

        if self.args.adversarial_weight > 0:
            self.optimizer_idx_to_mode = {0: 'gen', 1: 'gen', 2: 'dis'}


        if self.use_stylegan_d:
            if self.args.adversarial_weight > 0:
                self.optimizer_idx_to_mode = {0: 'gen', 1: 'gen',  2: 'dis',  3: 'dis_stylegan'}
            else:
                self.optimizer_idx_to_mode = {0: 'gen', 1: 'gen',  2: 'dis_stylegan'}


        OPTS = []
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
        OPTS.append(opt_gen)
        # OPTS.append(opt_gen)
        if self.args.adversarial_weight > 0:
            if self.args.use_second_dis:
                opt_dis = opts[self.args.dis_opt_type](
                    itertools.chain(self.discriminator.parameters(), self.discriminator2.parameters()),
                    # itertools.chain(self.discriminator.parameters()),
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
            OPTS.append(opt_dis)
        if self.use_stylegan_d:
            d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)
            opt_dis_style = opts['adam'](
                self.stylegan_discriminator.parameters(),
                self.args.dis_stylegan_lr * d_reg_ratio,
                0 ** d_reg_ratio,
                0.99 ** d_reg_ratio,
            )
            OPTS.append(opt_dis_style)


        return OPTS

    def configure_schedulers(self, opts, epochs=None, steps_per_epoch=None):
        SH = []
        ITERS = []
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
        SH.append(shd_gen)
        ITERS.append(self.args.gen_shd_max_iters)
        if self.args.adversarial_weight > 0:
            shd_dis = shds[self.args.dis_shd_type](
                opts[1],
                self.args.dis_lr,
                self.args.dis_shd_lr_min,
                self.args.dis_shd_max_iters,
                epochs,
                steps_per_epoch
            )
            SH.append(shd_dis)
            ITERS.append(self.args.dis_shd_max_iters)
        if self.use_stylegan_d:
            shd_dis_stylegan = shds[self.args.dis_shd_type](
                opts[-1],
                self.args.dis_stylegan_lr,
                self.args.dis_shd_lr_min,
                self.args.dis_shd_max_iters,
                epochs,
                steps_per_epoch
            )
            SH.append(shd_dis_stylegan)
            ITERS.append(self.args.dis_shd_max_iters)

        return SH, ITERS
