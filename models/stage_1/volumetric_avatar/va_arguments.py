from argparse import ArgumentParser
from utils import args as args_utils
import sys
sys.path.append('/fsx/nikitadrobyshev/')
from EmoPortraits.networks import basic_avatar, volumetric_avatar
from dataclasses import dataclass
from copy import deepcopy

class VolumetricAvatarConfig:
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
        parser.add_argument('--head_pose_regressor_path', default='/fsx/nikitadrobyshev/EmoPortraits/repos/head_pose_regressor.pth')
        parser.add_argument('--additive_motion', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--use_seg', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_back', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--features_sigm', default=1, type=int)

        parser.add_argument('--use_mix_mask', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_ibug_mask', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_old_fp', default='False', type=args_utils.str2bool, choices=[True, False])
        
        
        parser.add_argument('--use_masked_aug', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--resize_depth', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--volume_renderer_mode', default='depth_to_channels', type=str)

        parser.add_argument('--save_exp_vectors', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--aligned_warp_rot_source', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--aligned_warp_rot_target', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_right_3d_trans', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--use_unet_as_s_nw', default='False', type=args_utils.str2bool, choices=[True, False])

        

        parser.add_argument('--separate_stm', default='False', type=args_utils.str2bool, choices=[True, False])

        
        
        
        


        parser.add_argument('--init_type', default='kaiming')
        parser.add_argument('--init_gain', default=0.0, type=float)

        parser.add_argument('--sep_vol_loss_scale', default=0.0, type=float)

        

        parser.add_argument('--bs_resnet18_fv_mix', default=2, type=int)

        parser.add_argument('--dis_num_channels', default=64, type=int)
        parser.add_argument('--dis_max_channels', default=512, type=int)
        parser.add_argument('--dis_num_blocks', default=4, type=int)
        parser.add_argument('--dis_num_scales', default=1, type=int)

        parser.add_argument('--use_mix_dis', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dis2_num_channels', default=64, type=int)
        parser.add_argument('--dis2_max_channels', default=512, type=int)
        parser.add_argument('--dis2_num_blocks', default=4, type=int)
        parser.add_argument('--dis2_num_scales', default=2, type=int)
        parser.add_argument('--dis2_train_start', default=0, type=int)
        parser.add_argument('--dis2_gen_train_start', default=0, type=int)
        parser.add_argument('--dis2_gen_train_ratio', default=0, type=int)

        parser.add_argument('--dis_init_type', default='xavier')
        parser.add_argument('--dis_init_gain', default=0.02, type=float)

        parser.add_argument('--adversarial_weight', default=1.0, type=float)
        parser.add_argument('--mix_gen_adversarial', default=1.0, type=float)
        parser.add_argument('--feature_matching_weight', default=60.0, type=float)
        parser.add_argument('--vgg19_weight', default=20.0, type=float)
        parser.add_argument('--vgg19_neutral', default=0.0, type=float)
        parser.add_argument('--vgg19_neu_epoches', default=0, type=int)
        
        
        parser.add_argument('--vgg19_face', default=0.0, type=float)
        parser.add_argument('--perc_face_pars', default=0.0, type=float)
        parser.add_argument('--vgg19_face_mixing', default=0.0, type=float)
        parser.add_argument('--vgg19_fv_mix', default=0.0, type=float)
        parser.add_argument('--resnet18_fv_mix', default=0.0, type=float)
        parser.add_argument('--mix_losses_start', default=4, type=int)
        parser.add_argument('--contr_losses_start', default=1, type=int)

        parser.add_argument('--w_eyes_loss_l1', default=0.0, type=float)
        parser.add_argument('--w_mouth_loss_l1', default=0.0, type=float)
        parser.add_argument('--w_ears_loss_l1', default=0.0, type=float)
        

        parser.add_argument('--face_resnet', default=0.0, type=float)

        parser.add_argument('--vgg19_emotions', default=0.0, type=float)
        parser.add_argument('--resnet18_emotions', default=0.0, type=float)
        parser.add_argument('--landmarks', default=0.0, type=float)

        parser.add_argument('--l1_weight', default=0.0, type=float)
        parser.add_argument('--neu_exp_l1', default=0.0, type=float)
        parser.add_argument('--match_neutral', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--detach_lat_vol', default=-1, type=int)
        parser.add_argument('--freeze_proc_nw', default=-1, type=int)
        

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

        parser.add_argument('--volumes_pull', default=0.0, type=float)
        parser.add_argument('--volumes_push', default=0.0, type=float)
        
        parser.add_argument('--contrastive_exp', default=0.0, type=float)
        parser.add_argument('--contrastive_idt', default=0.0, type=float)

        parser.add_argument('--only_cycle_embed', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--detach_warp_mixing_embed', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--detach_dis_inputs', default='False', type=args_utils.str2bool, choices=[True, False])
        
        parser.add_argument('--unet_first', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_mix_losses', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--pred_cycle', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--expr_custom_w', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--detach_cv_l1', default='False', type=args_utils.str2bool, choices=[True, False])
        
        
        
        
        

        parser.add_argument('--gaze_weight', default=0.0, type=float)
        parser.add_argument('--vgg19_num_scales', default=4, type=int)
        parser.add_argument('--warping_reg_weight', default=0.0, type=float)

        parser.add_argument('--spn_networks',
                            default='local_encoder_nw, local_encoder_seg_nw, local_encoder_mask_nw, idt_embedder_nw, expression_embedder_nw, xy_generator_nw, uv_generator_nw, warp_embed_head_orig_nw, pose_embed_decode_nw, pose_embed_code_nw, volume_process_nw, volume_source_nw, volume_pred_nw, decoder_nw, backgroung_adding_nw, background_process_nw')
        parser.add_argument('--ws_networks',
                            default='local_encoder_nw, local_encoder_seg_nw, local_encoder_mask_nw, idt_embedder_nw, expression_embedder_nw, xy_generator_nw, uv_generator_nw, warp_embed_head_orig_nw,  pose_embed_decode_nw, pose_embed_code_nw, volume_process_nw, volume_source_nw, volume_pred_nw, decoder_nw, backgroung_adding_nw, background_process_nw')
        parser.add_argument('--spn_layers', default='conv2d, conv3d, linear, conv2d_ws, conv3d_ws')
        parser.add_argument('--use_sn', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_ws', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--print_norms', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--local_encoder_input_size', default=3, type=int)
        parser.add_argument('--no_channel_increase_3d_source', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--no_channel_increase_3d_pred', default='True', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--max_channel_res_3d_mul', default=4, type=int)

        parser.add_argument('--face_parts_epoch_start', default=10, type=int)
        
        

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
        parser.add_argument('--dis_input_channels', default=3, type=int)

        parser.add_argument('--dis_shd_type', default='cosine')
        parser.add_argument('--dis_shd_max_iters', default=2.5e5, type=int)
        parser.add_argument('--dis_shd_lr_min', default=4e-6, type=int)
        parser.add_argument('--eps', default=1e-8, type=float)


        # Gen parametres
        parser.add_argument('--gen_num_channels', default=32, type=int)
        parser.add_argument('--gen_max_channels', default=512, type=int)
        parser.add_argument('--dec_max_channels', default=512, type=int)
        parser.add_argument('--dec_no_detach_frec', default=2, type=int)
        parser.add_argument('--dec_key_emb', default='orig', type=str)
        

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
        parser.add_argument('--volume_rendering', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--l1_vol_rgb', default=0.0, type=float)        
        parser.add_argument('--l1_vol_rgb_mix', default=0.0, type=float)
        parser.add_argument('--start_vol_rgb', default=0, type=int)
        parser.add_argument('--squeeze_dim', default=0, type=int)
        parser.add_argument('--coarse_num_sample', default=48, type=int)
        parser.add_argument('--hidden_vol_dec_dim', default=448, type=int)
        parser.add_argument('--targ_vol_loss_scale', default=0.0, type=float)
        parser.add_argument('--num_layers_vol_dec', default=2, type=int)
        parser.add_argument('--vol_renderer_dec_channels', default=96*16, type=int)


        parser.add_argument('--volumes_l1', default=0.0, type=float)
        parser.add_argument('--predict_target_canon_vol', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--type_of_can_vol_loss', default='l1', type=str, choices=['l1', 'l2',])


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
        parser.add_argument('--emb_v_exp', default='False', type=args_utils.str2bool, choices=[True, False])

        
        parser.add_argument('--use_smart_scale', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--smart_scale_max_scale', default=0.75, type=float)
        parser.add_argument('--smart_scale_max_tol_angle', default=0.8, type=float)


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
        parser.add_argument('--dec_use_adanorm', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dec_use_adaconv', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dec_use_sg3_img_dec', default='False', type=args_utils.str2bool, choices=[True, False])
        

        
        parser.add_argument('--vol_loss_epoch', default=0, type=int)
        parser.add_argument('--vol_loss_grad', default=0, type=int)
        

        parser.add_argument('--dec_num_blocks', default=8, type=int)
        parser.add_argument('--dec_channel_mult', default=2.0, type=float)
        parser.add_argument('--dec_up_block_type', default='res', type=str, choices=['res', 'conv'])
        parser.add_argument('--gen_max_channels', default=512, type=int)
        parser.add_argument('--dec_pred_seg', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dec_seg_channel_mult', default=1.0, type=float)
        parser.add_argument('--im_dec_num_lrs_per_resolution', default=1, type=int)
        parser.add_argument('--im_dec_ch_div_factor', default=2, type=float)

        parser.add_argument('--dec_pred_conf', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--dec_bigger', default='False', type=args_utils.str2bool, choices=[True, False])

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
        parser.add_argument('--lpe_output_channels_expression', default=512, type=int)
        parser.add_argument('--exp_dropout', default=0.0, type=float)
        
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


    def __init__(self, args):
         self.args = args


    def get_weights(self):
        return {
            'barlow': self.args.barlow,
            'adversarial': self.args.adversarial_weight,
            'mix_gen_adversarial':self.args.mix_gen_adversarial,
            'feature_matching': self.args.feature_matching_weight,
            'vgg19': self.args.vgg19_weight,
            'vgg19_neutral': self.args.vgg19_neutral,
            'warping_reg': self.args.warping_reg_weight,
            'gaze': self.args.gaze_weight,
            'l1_weight': self.args.l1_weight,
            'l1_back': self.args.l1_back,
            'vgg19_face': self.args.vgg19_face,
            'perc_face_pars': self.args.perc_face_pars,
            'vgg19_face_mixing': self.args.vgg19_face_mixing,
            'resnet18_fv_mix': self.args.resnet18_fv_mix,
            'vgg19_fv_mix': self.args.vgg19_fv_mix,

            'face_resnet': self.args.face_resnet,
            'vgg19_emotions': self.args.vgg19_emotions,
            'landmarks': self.args.landmarks,
            'resnet18_emotions': self.args.resnet18_emotions,
            'cycle_idn': self.args.cycle_idn,
            'cycle_exp': self.args.cycle_exp,
            'l1_vol_rgb':self.args.l1_vol_rgb,
            'l1_vol_rgb_mix':self.args.l1_vol_rgb_mix,
            'volumes_l1':self.args.volumes_l1,
            'vgg19_face_cycle_idn': self.args.vgg19_face_cycle_idn,
            'vgg19_cycle_idn': self.args.vgg19_weight_cycle_idn,

            'vgg19_face_cycle_exp': self.args.vgg19_face_cycle_exp,
            'vgg19_cycle_exp': self.args.vgg19_weight_cycle_exp,

            'stm':self.args.stm,
            'pull_idt': self.args.pull_idt,
            'pull_exp': self.args.pull_exp,
            'volumes_pull': self.args.volumes_pull,
            'volumes_push': self.args.volumes_push,
            'push_idt': self.args.push_idt,
            'push_exp': self.args.push_exp,
            'contrastive_exp': self.args.contrastive_exp,
            'contrastive_idt': self.args.contrastive_idt,
            'stylegan_weight': self.args.stylegan_weight,
            'neutral_expr_l1_weight': self.args.neu_exp_l1
        }
    

    @property
    def unet3d_cfg(self) -> volumetric_avatar.Unet3D.Config:
        return volumetric_avatar.Unet3D.Config(
            eps=self.args.eps,
            num_gpus=self.args.num_gpus,
            gen_embed_size=self.args.gen_embed_size,
            gen_adaptive_kernel=self.args.gen_adaptive_kernel,
            gen_use_adanorm=self.args.gen_use_adanorm,
            gen_use_adaconv=self.args.gen_use_adaconv,
            gen_upsampling_type=self.args.gen_upsampling_type,
            gen_downsampling_type=self.args.gen_downsampling_type,
            gen_dummy_input_size=self.args.gen_dummy_input_size,
            gen_latent_texture_size=self.args.gen_latent_texture_size,
            gen_latent_texture_depth=self.args.gen_latent_texture_depth,
            gen_adaptive_conv_type=self.args.gen_adaptive_conv_type,
            gen_latent_texture_channels=self.args.gen_latent_texture_channels,
            gen_activation_type=self.args.gen_activation_type,
            gen_max_channels=self.args.gen_max_channels_unet3d,
            warp_norm_grad=self.args.warp_norm_grad,
            warp_block_type=self.args.warp_block_type,
            tex_pred_rgb=self.args.tex_pred_rgb,
            image_size=self.args.image_size,
            tex_use_skip_resblock=self.args.tex_use_skip_resblock,
            norm_layer_type=self.args.norm_layer_type,
        )

    @property
    def unet3d_cfg_s(self) -> volumetric_avatar.Unet3D.Config:
        return volumetric_avatar.Unet3D.Config(
            eps=self.args.eps,
            num_gpus=self.args.num_gpus,
            gen_embed_size=self.args.gen_embed_size,
            gen_adaptive_kernel=self.args.gen_adaptive_kernel,
            gen_use_adanorm=self.args.gen_use_adanorm,
            gen_use_adaconv=self.args.gen_use_adaconv,
            gen_upsampling_type=self.args.gen_upsampling_type,
            gen_downsampling_type=self.args.gen_downsampling_type,
            gen_dummy_input_size=16,
            gen_latent_texture_size=self.args.gen_latent_texture_size,
            gen_latent_texture_depth=self.args.gen_latent_texture_depth,
            gen_adaptive_conv_type=self.args.gen_adaptive_conv_type,
            gen_latent_texture_channels=self.args.gen_latent_texture_channels,
            gen_activation_type=self.args.gen_activation_type,
            gen_max_channels=96+32,
            warp_norm_grad=self.args.warp_norm_grad,
            warp_block_type=self.args.warp_block_type,
            tex_pred_rgb=self.args.tex_pred_rgb,
            image_size=self.args.image_size,
            tex_use_skip_resblock=self.args.tex_use_skip_resblock,
            norm_layer_type=self.args.norm_layer_type,
        )



    @property
    def local_encoder_cfg(self) -> volumetric_avatar.LocalEncoder.Config:
        return volumetric_avatar.LocalEncoder.Config(
            num_gpus=self.args.num_gpus,
            gen_upsampling_type=self.args.gen_upsampling_type,
            gen_downsampling_type=self.args.gen_downsampling_type,
            gen_input_image_size=self.args.image_size,
            gen_latent_texture_size=self.args.latent_volume_size,
            gen_latent_texture_depth=self.args.latent_volume_depth,
            gen_num_channels=self.args.gen_num_channels,
            enc_channel_mult=self.args.enc_channel_mult,
            norm_layer_type=self.args.norm_layer_type,
            gen_max_channels=self.args.gen_max_channels,
            enc_block_type=self.args.enc_block_type,
            gen_activation_type=self.args.gen_activation_type,
            gen_latent_texture_channels=self.args.latent_volume_channels,
            in_channels=self.args.local_encoder_input_size,
            warp_norm_grad=self.args.warp_norm_grad,
        )
    
    @property
    def local_encoder_back_cfg(self) -> volumetric_avatar.LocalEncoderBack.Config:
        return volumetric_avatar.LocalEncoderBack.Config(
                gen_upsampling_type=self.args.gen_upsampling_type,
                gen_downsampling_type=self.args.gen_downsampling_type,
                gen_num_channels=self.args.gen_num_channels,
                enc_channel_mult=self.args.enc_channel_mult,
                norm_layer_type=self.args.norm_layer_type,
                num_gpus=self.args.num_gpus,
                gen_input_image_size=self.args.image_size,
                gen_latent_texture_size=self.args.gen_latent_texture_size,
                gen_max_channels=self.args.gen_max_channels,
                enc_block_type=self.args.enc_block_type,
                gen_activation_type=self.args.gen_activation_type,
                seg_out_channels=self.args.background_net_input_channels,
                in_channels=self.args.local_encoder_input_size,
        )

    @property
    def volume_renderer_cfg(self) -> volumetric_avatar.VolumeRenderer.Config:
        return volumetric_avatar.VolumeRenderer.Config(
        dec_channels= self.args.vol_renderer_dec_channels,
        img_channels=self.args.dec_max_channels,
        squeeze_dim=self.args.squeeze_dim,
        features_sigm=self.args.features_sigm,
        depth_resolution=self.args.coarse_num_sample,
        hidden_vol_dec_dim=self.args.hidden_vol_dec_dim,
        num_layers_vol_dec=self.args.num_layers_vol_dec)

    @property
    def idt_embedder_cfg(self) -> volumetric_avatar.IdtEmbed.Config:
         return volumetric_avatar.IdtEmbed.Config(
            idt_backbone=self.args.idt_backbone,
            num_source_frames=self.args.num_source_frames,
            idt_output_size=self.args.idt_output_size,
            idt_output_channels=self.args.idt_output_channels,
            num_gpus=self.args.num_gpus,
            norm_layer_type=self.args.norm_layer_type,
            idt_image_size=self.args.idt_image_size
         )
    
    @property
    def exp_embedder_cfg(self) -> volumetric_avatar.ExpressionEmbed.Config:
         return volumetric_avatar.ExpressionEmbed.Config(
            lpe_head_backbone=self.args.lpe_head_backbone,
            lpe_face_backbone=self.args.lpe_face_backbone,
            image_size=self.args.exp_image_size,
            project_dir=self.args.project_dir,
            num_gpus=self.args.num_gpus,
            lpe_output_channels=self.args.lpe_output_channels,
            lpe_output_channels_expression=self.args.lpe_output_channels_expression,
            lpe_final_pooling_type=self.args.lpe_final_pooling_type,
            lpe_output_size=self.args.lpe_output_size,
            lpe_head_transform_sep_scales=self.args.lpe_head_transform_sep_scales,
            norm_layer_type=self.args.norm_layer_type,
            use_smart_scale = self.args.use_smart_scale,
            smart_scale_max_scale = self.args.smart_scale_max_scale,
            smart_scale_max_tol_angle = self.args.smart_scale_max_tol_angle,
            dropout=self.args.exp_dropout,
            custom_w=self.args.expr_custom_w
         )
    
    @property
    def warp_generator_cfg(self) -> volumetric_avatar.WarpGenerator.Config:
         return volumetric_avatar.WarpGenerator.Config(
            eps=self.args.eps,
            num_gpus=self.args.num_gpus,
            gen_adaptive_conv_type=self.args.gen_adaptive_conv_type,
            gen_activation_type=self.args.gen_activation_type,
            gen_upsampling_type=self.args.gen_upsampling_type,
            gen_downsampling_type=self.args.gen_downsampling_type,
            gen_dummy_input_size=self.args.gen_embed_size,
            gen_latent_texture_depth=self.args.gen_latent_texture_depth,
            gen_latent_texture_size=self.args.gen_latent_texture_size,
            gen_max_channels=self.args.gen_max_channels,
            gen_num_channels=self.args.gen_num_channels,
            gen_use_adaconv=self.args.gen_use_adaconv,
            gen_adaptive_kernel=self.args.gen_adaptive_kernel,
            gen_embed_size=self.args.gen_embed_size,
            warp_output_size=self.args.warp_output_size,
            warp_channel_mult=self.args.warp_channel_mult,
            warp_block_type=self.args.warp_block_type,
            norm_layer_type=self.args.norm_layer_type,
            input_channels=self.args.gen_max_channels
         )
    
    def get_channel_mul_for_res_3D(self, n, N_blocks, max_mul):
        # To get pattern like (1, 2, 4, 4, 2, 1) if N_blocks == 6
        # and (1, 2, 4, 8, 4, 2, 1) if N_blocks == 7

        if n<(N_blocks+1)//2:
            curr_mul = min(2**n, max_mul)
        else:
            curr_mul = 2**(N_blocks-(n+1))

        return int(curr_mul)

    @property
    def VPN_resblocks_source_cfg(self) -> volumetric_avatar.VPN_ResBlocks.Config:

        return volumetric_avatar.VPN_ResBlocks.Config(
            num_gpus=self.args.num_gpus,
            norm_layer_type=self.args.norm_layer_type,
            input_channels=self.args.latent_volume_channels,
            num_blocks=self.args.source_volume_num_blocks,
            activation_type=self.args.gen_activation_type,
            conv_layer_type='conv_3d',
            channels=[] if self.args.no_channel_increase_3d_source else [self.get_channel_mul_for_res_3D(i, self.args.source_volume_num_blocks, self.args.max_channel_res_3d_mul)*self.args.latent_volume_channels 
                                                                  for i in range(self.args.source_volume_num_blocks)]
            )
    
    @property
    def VPN_resblocks_pred_cfg(self) -> volumetric_avatar.VPN_ResBlocks.Config:
        return volumetric_avatar.VPN_ResBlocks.Config(
            num_gpus=self.args.num_gpus,
            norm_layer_type=self.args.norm_layer_type,
            input_channels=self.args.latent_volume_channels,
            num_blocks=self.args.pred_volume_num_blocks,
            activation_type=self.args.gen_activation_type,
            conv_layer_type='conv_3d',
            channels=[] if self.args.no_channel_increase_3d_pred else [self.get_channel_mul_for_res_3D(i, self.args.pred_volume_num_blocks, self.args.max_channel_res_3d_mul)*self.args.latent_volume_channels 
                                                                  for i in range(self.args.pred_volume_num_blocks)]
            )
    
    @property
    def decoder_cfg(self) -> volumetric_avatar.Decoder.Config:
        return volumetric_avatar.Decoder.Config(
            eps=self.args.eps,
            image_size=self.args.image_size,
            gen_embed_size=self.args.gen_embed_size,
            gen_adaptive_kernel=self.args.gen_adaptive_kernel,
            gen_adaptive_conv_type=self.args.gen_adaptive_conv_type,
            gen_latent_texture_size=self.args.gen_latent_texture_size,
            in_channels=self.args.gen_latent_texture_channels * self.args.gen_latent_texture_depth,
            gen_num_channels=self.args.gen_num_channels,
            dec_max_channels=self.args.dec_max_channels,
            gen_use_adanorm=self.args.dec_use_adanorm,
            gen_activation_type=self.args.gen_activation_type,
            gen_use_adaconv=self.args.dec_use_adaconv,
            dec_channel_mult=self.args.dec_channel_mult,
            dec_num_blocks=self.args.dec_num_blocks,
            dec_up_block_type=self.args.dec_up_block_type,
            dec_pred_seg=self.args.dec_pred_seg,
            dec_seg_channel_mult=self.args.dec_seg_channel_mult,
            num_gpus=self.args.num_gpus,
            norm_layer_type=self.args.norm_layer_type,
            bigger=self.args.dec_bigger,
            vol_render=self.args.volume_rendering,
            im_dec_num_lrs_per_resolution = self.args.im_dec_num_lrs_per_resolution,
            im_dec_ch_div_factor = self.args.im_dec_ch_div_factor,
            emb_v_exp = self.args.emb_v_exp,
            dec_use_sg3_img_dec = self.args.dec_use_sg3_img_dec,
            no_detach_frec = self.args.dec_no_detach_frec,
            dec_key_emb = self.args.dec_key_emb

        )
    
    @property
    def dis_cfg(self) -> basic_avatar.MultiScaleDiscriminator.Config:
        return basic_avatar.MultiScaleDiscriminator.Config(
                min_channels=self.args.dis_num_channels,
                max_channels=self.args.dis_max_channels,
                num_blocks=self.args.dis_num_blocks,
                input_channels=self.args.dis_input_channels,
                input_size=self.args.image_size,
                num_scales=self.args.dis_num_scales
            )
    
    @property
    def dis_2_cfg(self) -> basic_avatar.MultiScaleDiscriminator.Config:
        return basic_avatar.MultiScaleDiscriminator.Config(
                 min_channels=self.args.dis2_num_channels,
                    max_channels=self.args.dis2_max_channels,
                    num_blocks=self.args.dis2_num_blocks,
                    input_channels=self.args.dis_input_channels,
                    input_size=self.args.image_size,
                    num_scales=self.args.dis2_num_scales
            )
    

