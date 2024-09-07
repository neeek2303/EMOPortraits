import torch
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
import numpy as np
import itertools
from torch.cuda import amp
# sys.path.append('/fsx/nikitadrobyshev/')
from networks import basic_avatar, volumetric_avatar
from utils import args as args_utils
from utils import spectral_norm, weight_init, point_transforms
from skimage.measure import label
from .va_losses_and_visuals import calc_train_losses, calc_test_losses, prepare_input_data, MODNET, init_losses
from .va_losses_and_visuals import visualize_data, get_visuals, draw_stickman
from .va_arguments import VolumetricAvatarConfig
from networks.volumetric_avatar.utils import requires_grad, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, \
    _calc_r1_penalty
from scipy import linalg
from dataclasses import dataclass
from torch.autograd import Variable
import math
from utils.non_specific import calculate_obj_params, FaceParsingBUG, get_mixing_theta, align_keypoints_torch

from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap
from ibug.roi_tanh_warping import roi_tanh_polar_restore, roi_tanh_polar_warp
from torchvision import transforms
from .va_arguments import VolumetricAvatarConfig
from utils import point_transforms



to_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

import contextlib

class Model(nn.Module):

    def __init__(self, args, training=True, rank=0, exp_dir=None):
        super(Model, self).__init__()
        self.args = args
        self.exp_dir = exp_dir
        print(self.exp_dir)
        self.va_config = VolumetricAvatarConfig(args)
        self.weights = self.va_config.get_weights()


        
        self.rank=rank
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


        # if self.args.w_eyes_loss_l1>0 or self.args.w_mouth_loss_l1>0 or self.args.w_ears_loss_l1>0:
        self.face_parsing_bug = FaceParsingBUG()


        self.m_key_diff = 0
        self.init_networks(args, training)


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
        grid_s = torch.linspace(-1, 1, self.args.aug_warp_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('identity_grid_2d', torch.stack([u, v], dim=2).view(1, -1, 2), persistent=False)

        grid_s = torch.linspace(-1, 1, self.args.latent_volume_size)
        grid_z = torch.linspace(-1, 1, self.args.latent_volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer('identity_grid_3d', torch.stack([u, v, w, e], dim=3).view(1, -1, 4), persistent=False)
        self.only_cycle_embed = args.only_cycle_embed

        self.use_masked_aug = args.use_masked_aug
        self.num_b_negs = self.args.num_b_negs
        self.pred_cycle = args.pred_cycle

        # # Apply spectral norm
        if args.use_sn:
            spectral_norm.apply_sp_to_nets(self)

        # Apply weight standartization 
        if args.use_ws:
            volumetric_avatar.utils.apply_ws_to_nets(self)

        # Calculate params
        calculate_obj_params(self)

        if training:
            self.init_losses(args)

    def init_networks(self, args, training):
        
        ##################################
        #### Encoders ####
        ##################################

        ## Define image encoder
        self.local_encoder_nw = volumetric_avatar.LocalEncoder(self.va_config.local_encoder_cfg)

        ## Define background nets; default = False
        if self.args.use_back:
            in_u = self.args.background_net_input_channels
            c = self.args.latent_volume_channels
            d = self.args.latent_volume_depth
            self.background_net_out_channels = self.args.latent_volume_depth * self.args.latent_volume_channels
            u = self.background_net_out_channels
            
            self.local_encoder_back_nw = volumetric_avatar.LocalEncoderBack(self.va_config.local_encoder_back_cfg)
            
            self.backgroung_adding_nw = nn.Sequential(*[nn.Conv2d(
                in_channels=c * d + u,
                out_channels=c * d,
                kernel_size=(1, 1),
                padding=0,
                bias=False),
                nn.ReLU(),
            ])

            self.background_process_nw = volumetric_avatar.UNet(in_u, u, base=self.args.back_unet_base, max_ch=self.args.back_unet_max_ch, norm='gn')

        # Define volume rendering net; default = False
        if self.args.volume_rendering:
            self.volume_renderer_nw = volumetric_avatar.VolumeRenderer(self.va_config.volume_renderer_cfg)

        # Define idt embedder net - for adaptivity of networks and for face warping help
        self.idt_embedder_nw = volumetric_avatar.IdtEmbed(self.va_config.idt_embedder_cfg)


        # Define expression embedder net - derive latent vector of emotions
        self.expression_embedder_nw = volumetric_avatar.ExpressionEmbed(self.va_config.exp_embedder_cfg)

        ##################################
        #### Warp ####
        ##################################

        # Operator that transform exp_emb to extended exp_emb (to match idt_emb size)
        self.pose_unsqueeze_nw = nn.Linear(args.lpe_output_channels_expression, args.gen_max_channels * self.embed_size ** 2,
                                        bias=False)


        # Operator that combine idt_imb and extended exp_emb together (a "+" sign of a scheme)
        self.warp_embed_head_orig_nw = nn.Conv2d(
            in_channels=args.gen_max_channels*(2 if self.args.cat_em else 1),
            out_channels=args.gen_max_channels,
            kernel_size=(1, 1),
            bias=False)
        
        # Define networks from warping to (xy) and from (uv) canical volume cube
        self.xy_generator_nw = volumetric_avatar.WarpGenerator(self.va_config.warp_generator_cfg)
        self.uv_generator_nw = volumetric_avatar.WarpGenerator(self.va_config.warp_generator_cfg)

        ##################################
        #### Volume process ####
        ##################################

        ## Define 3D net that goes right after image encoder
        if self.args.source_volume_num_blocks>0:
            
            if self.args.unet_first:
                print('aaaaaaaaaaaaaaaa')
                self.volume_source_nw = volumetric_avatar.Unet3D(self.va_config.unet3d_cfg_s)
            else:
                print('bbbbbbbbbbb')
                # self.volume_source_nw = volumetric_avatar.Unet3D(self.va_config.unet3d_cfg_s)
                self.volume_source_nw = volumetric_avatar.VPN_ResBlocks(self.va_config.VPN_resblocks_source_cfg)



        # If we want to use additional learnable tensor - like avarage person
        if self.args.use_tensor:
            d = self.args.gen_latent_texture_depth
            s = self.args.gen_latent_texture_size
            c = self.args.gen_latent_texture_channels
            self.avarage_tensor_ts = nn.Parameter(Variable(  (torch.rand((1,c,d,s,s), requires_grad = True)*2 - 1)*math.sqrt(6./(d*s*s*c))  ).cuda(), requires_grad = True)

        # Net that process volume after first duble-warping
        self.volume_process_nw = volumetric_avatar.Unet3D(self.va_config.unet3d_cfg)


        ## Define 3D net that goes before image decoder
        if self.args.pred_volume_num_blocks>0:
            
            if self.args.unet_first:
                self.volume_pred_nw = volumetric_avatar.Unet3D(self.va_config.unet3d_cfg_s)
            else:
                self.volume_pred_nw = volumetric_avatar.VPN_ResBlocks(self.va_config.VPN_resblocks_pred_cfg)

        ##################################
        #### Decoding ####
        ##################################
        self.decoder_nw = volumetric_avatar.Decoder(self.va_config.decoder_cfg)


            
        ##################################
        #### Discriminators ####
        ##################################
        if training:
            self.discriminator_ds = basic_avatar.MultiScaleDiscriminator(self.va_config.dis_cfg)
            self.discriminator_ds.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))

            if self.args.use_mix_dis:
                self.discriminator2_ds = basic_avatar.MultiScaleDiscriminator(self.va_config.dis_2_cfg)
                self.discriminator2_ds.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))

            if self.use_stylegan_d:
                self.r1_loss = torch.tensor(0.0)
                self.stylegan_discriminator_ds = basic_avatar.DiscriminatorStyleGAN2(size=self.args.image_size,
                                                                                  channel_multiplier=1, my_ch=2)

        
        ###########################################
        #### Non-trainable additional networks ####
        ###########################################
        self.face_idt = volumetric_avatar.FaceParsing(None, 'cuda', project_dir = self.args.project_dir)



        if self.args.use_mix_losses or self.pred_seg:
            self.get_mask = MODNET(project_dir = self.args.project_dir)

        if self.args.estimate_head_pose_from_keypoints:
            self.head_pose_regressor = volumetric_avatar.HeadPoseRegressor(args.head_pose_regressor_path, args.num_gpus)


        if args.warp_norm_grad:
            self.grid_sample = volumetric_avatar.GridSample(args.gen_latent_texture_size)
        else:
            self.grid_sample = lambda inputs, grid: F.grid_sample(inputs.float(), grid.float(),
                                                                  padding_mode=self.args.grid_sample_padding_mode)

        self.get_face_vector = volumetric_avatar.utils.Face_vector(self.head_pose_regressor, half=False)
        self.get_face_vector_resnet = volumetric_avatar.utils.Face_vector_resnet(half=False, project_dir=self.args.project_dir)
        self.face_parsing_bug = FaceParsingBUG()

        grid_s = torch.linspace(-1, 1, self.args.aug_warp_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('identity_grid_2d', torch.stack([u, v], dim=2).view(1, -1, 2), persistent=False)

        grid_s = torch.linspace(-1, 1, self.args.latent_volume_size)
        grid_z = torch.linspace(-1, 1, self.args.latent_volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer('identity_grid_3d', torch.stack([u, v, w, e], dim=3).view(1, -1, 4), persistent=False)



    def init_losses(self, args):
        return init_losses(self, args)


    def G_forward(self, data_dict, visualize, iteration=0, epoch=0):
        self.visualize = visualize


        b = data_dict['source_img'].shape[0]
        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth

        # If use face parsing mask 
        if self.args.use_mix_mask:
            trashhold = 0.6
            if self.args.use_ibug_mask:
                if not self.args.use_old_fp:
                    try:
                        face_mask_source_list = []
                        face_mask_target_list = []
                        for i in range(data_dict['source_img'].shape[0]):
                            masks_gt, logits_gt, logits_source_soft, faces = self.face_parsing_bug.get_lips(data_dict['source_img'][i])
                            masks_s1, logits_s1, logits_target_soft, _ = self.face_parsing_bug.get_lips(data_dict['target_img'][i])

                            logits_source_soft = logits_source_soft.detach()
                            logits_target_soft = logits_target_soft.detach()
                            
                            desired_indexes = [0,]

                            face_mask_source = 0
                            face_mask_target = 0
                            for i in desired_indexes: 
                                face_mask_source += logits_source_soft[:, i:i+1]
                                face_mask_target += logits_target_soft[:, i:i+1]
                            face_mask_source_list.append(face_mask_source)
                            face_mask_target_list.append(face_mask_target)
                            
                        face_mask_source = torch.cat(face_mask_source_list, dim=0)
                        face_mask_target = torch.cat(face_mask_target_list, dim=0)
                    except Exception as e:
                        print(e)
                        _, face_mask_source, _, cloth_s = self.face_idt.forward(data_dict['source_img'])
                        _, face_mask_target, _, cloth_t = self.face_idt.forward(data_dict['target_img'])
                else:
                        _, face_mask_source, _, cloth_s = self.face_idt.forward(data_dict['source_img'])
                        _, face_mask_target, _, cloth_t = self.face_idt.forward(data_dict['target_img'])

                _, _, hat_s, cloth_s = self.face_idt.forward(data_dict['source_img'])
                _, _, hat_t, cloth_t = self.face_idt.forward(data_dict['target_img'])

                face_mask_source+=hat_s
                face_mask_target+=hat_t


                data_dict['source_mask_modnet'] = data_dict['source_mask'].clone()
                data_dict['target_mask_modnet'] = data_dict['target_mask'].clone()
                data_dict['source_mask_modnet'][:, :, -256:]*=0
                data_dict['target_mask_modnet'][:, :, -256:]*=0
                face_mask_source = (face_mask_source+data_dict['source_mask_modnet'] >= trashhold).float()
                face_mask_target = (face_mask_target+data_dict['target_mask_modnet'] >= trashhold).float()
                data_dict['source_mask_face_pars_1'] = (face_mask_source).float()
                data_dict['target_mask_face_pars_1'] = (face_mask_target).float()
                data_dict['source_mask'] = (data_dict['source_mask'] * data_dict['source_mask_face_pars_1']).float()
                data_dict['target_mask'] = (data_dict['target_mask'] * data_dict['target_mask_face_pars_1']).float()
                
            else:
                face_mask_source, _, _, cloth_s = self.face_idt.forward(data_dict['source_img'])
                face_mask_target, _, _, cloth_t = self.face_idt.forward(data_dict['target_img'])
                face_mask_source = (face_mask_source > trashhold).float()
                face_mask_target = (face_mask_target > trashhold).float()

                data_dict['source_mask_modnet'] = data_dict['source_mask']
                data_dict['target_mask_modnet'] = data_dict['target_mask']
                data_dict['source_mask_face_pars'] = (face_mask_source).float()
                data_dict['target_mask_face_pars'] = (face_mask_target).float()

                data_dict['source_mask'] = (data_dict['source_mask'] * data_dict['source_mask_face_pars']).float()
                data_dict['target_mask'] = (data_dict['target_mask'] * data_dict['target_mask_face_pars']).float()



        c = self.args.latent_volume_channels
        s = self.args.latent_volume_size
        d = self.args.latent_volume_depth

        # if iteration==200:
        #     print('aaaaaaaa')
        #     print(self.m_key_diff/200)


        # Estimate head rotation and corresponded warping
        if self.args.estimate_head_pose_from_keypoints:
            with torch.no_grad():
                data_dict['source_theta'], source_scale, data_dict['source_rotation'], source_tr = self.head_pose_regressor.forward(data_dict['source_img'], return_srt=True)
                data_dict['target_theta'], target_scale, data_dict['target_rotation'], target_tr = self.head_pose_regressor.forward(data_dict['target_img'], return_srt=True)

            grid = self.identity_grid_3d.repeat_interleave(b, dim=0)

            inv_source_theta = data_dict['source_theta'].float().inverse().type(data_dict['source_theta'].type())

            data_dict['source_rotation_warp'] = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3) # транспонируем так как вектора слева (grid) и транспонирован

            data_dict['source_warped_keypoints'] = data_dict['source_keypoints'].bmm(inv_source_theta[:, :3, :3])


            data_dict['source_warped_keypoints_n'] = data_dict['source_warped_keypoints'].clone()
            data_dict['source_warped_keypoints_n'][:, 27:31] = torch.tensor([[-0.0000, -0.2,  0.22],
                                                                           [-0.0000, -0.13,  0.26],
                                                                           [-0.0000, -0.06,  0.307],
                                                                           [-0.0000, -0.008,  0.310]]).to(data_dict['source_warped_keypoints'].device)
            

            # data_dict['source_warped_keypoints_n'][:, 27:31] = torch.tensor([[-0.0000, -0.30,  0.20],
            #                                                                [-0.0000, -0.20,  0.25],
            #                                                                [-0.0000, -0.10,  0.300],
            #                                                                [-0.0000, -0.05,  0.295]]).to(data_dict['source_warped_keypoints'].device)


            data_dict['source_warped_keypoints_n'], transform_matrix_s = align_keypoints_torch(data_dict['source_warped_keypoints_n'], data_dict['source_warped_keypoints'], nose=True)


            transform_matrix_s = transform_matrix_s.to(data_dict['source_rotation_warp'].device)

            new_m = inv_source_theta[:, :3, :3].bmm(transform_matrix_s[:, :3, :3])
            data_dict['source_warped_keypoints_n'] = data_dict['source_keypoints'].bmm(new_m)
            data_dict['source_warped_keypoints_n']+=transform_matrix_s[:, None, :3, 3]


            if self.args.aligned_warp_rot_source:
                new_m_warp_s = inv_source_theta.bmm(transform_matrix_s)
                data_dict['source_rotation_warp'] = grid.bmm(new_m_warp_s[:, :3].transpose(1, 2))
                # data_dict['source_rotation_warp']+= transform_matrix_s[:, None, :3, 3]
                data_dict['source_rotation_warp'] = data_dict['source_rotation_warp'].view(-1, d, s, s, 3)
                
            else:
                data_dict['source_rotation_warp'] = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3) # транспонируем так как вектора слева (grid) и транспонирован




            if self.args.aligned_warp_rot_target:
                inv_transform_matrix  = transform_matrix_s.float().inverse().type(data_dict['target_theta'].type())
                new_m_warp_t = inv_transform_matrix.bmm(data_dict['target_theta'])
                data_dict['target_rotation_warp'] = grid.bmm(new_m_warp_t[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
                data_dict['target_pre_warped_keypoints'] = data_dict['source_warped_keypoints_n'].bmm(inv_transform_matrix[:, :3, :3])
                data_dict['target_warped_keypoints'] = data_dict['target_pre_warped_keypoints'].bmm(data_dict['target_theta'][:, :3, :3])
            else:
                data_dict['target_rotation_warp'] = grid.bmm(data_dict['target_theta'][:, :3].transpose(1, 2)).view(-1, d,s, s, 3)




            if self.args.predict_target_canon_vol and not (epoch==0 and iteration<0):
                theta_st = point_transforms.get_transform_matrix(source_scale, data_dict['target_rotation'], target_tr)
                inv_target_theta = theta_st.float().inverse().type(data_dict['target_theta'].type())
                
                data_dict['target_warped_keypoints'] = data_dict['target_keypoints'].bmm(inv_target_theta[:, :3, :3])


                data_dict['target_warped_keypoints_aligned'], transform_matrix = align_keypoints_torch(data_dict['source_warped_keypoints'], data_dict['target_warped_keypoints'])

                transform_matrix = transform_matrix.to(data_dict['target_keypoints'])
                # data_dict['target_warped_keypoints_aligned_b'] = data_dict['target_warped_keypoints'].bmm(transform_matrix[:, :3, :3].transpose(1, 2))
                # data_dict['target_warped_keypoints_aligned_b']+=transform_matrix[:, None, :3, 3]

                new_m_warp_s = inv_target_theta.bmm(transform_matrix)
                data_dict['target_inv_rotation_warp'] = grid.bmm(new_m_warp_s[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
                # data_dict['target_inv_rotation_warp'] = data_dict['target_inv_rotation_warp'].bmm(transform_matrix[:, :3, :3].transpose(1, 2))
                # data_dict['target_inv_rotation_warp']+= transform_matrix[:, None, :3, 3]
                # data_dict['target_inv_rotation_warp'] = data_dict['target_inv_rotation_warp'].view(-1, d, s, s, 3)
                

        else:
            data_dict['source_rotation_warp'] = point_transforms.world_to_camera(self.identity_grid_3d[..., :3],
                                                                                 data_dict['source_params_3dmm']).view(
                b, d, s, s, 3)
            data_dict['target_rotation_warp'] = point_transforms.camera_to_world(self.identity_grid_3d[..., :3],
                                                                                 data_dict['target_params_3dmm']).view(
                b, d, s, s, 3)


        # Local encoder
        latent_volume = self.local_encoder_nw(data_dict['source_img'] * data_dict['source_mask'])

        # Idt and expression vectors
        data_dict['idt_embed'] = self.idt_embedder_nw(data_dict['source_img'] * data_dict['source_mask'])
        data_dict = self.expression_embedder_nw(data_dict, self.args.estimate_head_pose_from_keypoints,
                                             self.use_masked_aug)

        # Produce embeddings for warping
        source_warp_embed_dict, target_warp_embed_dict, mixing_warp_embed_dict, embed_dict = self.predict_embed(
            data_dict)


        # Predict a warping from image features to texture inputs
        xy_gen_warp, data_dict['source_delta_xy'] = self.xy_generator_nw(source_warp_embed_dict)
        source_xy_warp_resize = xy_gen_warp
        if self.resize_warp:
            source_xy_warp_resize = self.resize_warp_func(source_xy_warp_resize)  # if warp is predicted at higher resolution than resolution of texture

        # Predict warping of texture according to target pose
        target_uv_warp, data_dict['target_delta_uv'] = self.uv_generator_nw(target_warp_embed_dict)
        target_uv_warp_resize = target_uv_warp
        if self.resize_warp:
            target_uv_warp_resize = self.resize_warp_func(target_uv_warp_resize) # if warp is predicted at higher resolution than resolution of texture

        # Finally two warpings
        data_dict['source_xy_warp_resize'] = source_xy_warp_resize
        data_dict['target_uv_warp_resize'] = target_uv_warp_resize

        # Adding back features to face features, # default: False, as we don't use background net
        if self.args.use_back: 
            seg_encod_input = data_dict['source_img'] * (1 - data_dict['source_mask'])
            source_latents_background = self.local_encoder_back_nw(seg_encod_input)
            target_latent_feats_back = self.background_process_nw(source_latents_background)

        # Reshape latents into 3D volume
        latent_volume = latent_volume.view(b, c, d, s, s)
        

        # features 3D prepocess
        if self.args.unet_first:
            latent_volume = self.volume_process_nw(latent_volume, embed_dict)
        else:
            if self.args.source_volume_num_blocks > 0:
                latent_volume = self.volume_source_nw(latent_volume)

###############################################
        if self.args.detach_lat_vol>0:
            if iteration%self.args.detach_lat_vol==0:
                latent_volume = latent_volume.detach()


        if self.args.freeze_proc_nw>0:
            if iteration%self.args.freeze_proc_nw==0:
                for param in self.volume_process_nw.parameters():
                    param.requires_grad = False
            else:
                for param in self.volume_process_nw.parameters():
                    param.requires_grad = True 



        # Warp from source pose
        latent_volume = self.grid_sample(
            self.grid_sample(latent_volume, data_dict['source_rotation_warp']),
            data_dict['source_xy_warp_resize'])

        # Process latent texture
        if self.args.unet_first:
            if self.args.source_volume_num_blocks > 0:
                canonical_latent_volume = self.volume_source_nw(latent_volume)
        else:
            canonical_latent_volume = self.volume_process_nw(latent_volume, embed_dict)





        if self.args.use_tensor:
            canonical_latent_volume+=self.avarage_tensor_ts



        # # Features 3D prepocess after adding avrg tensor
        # if self.args.pred_volume_num_blocks > 0:
        #     canonical_latent_volume = self.volume_pred_nw(canonical_latent_volume)


        # if want to apply self-supervise learning for canonical tensor
        if self.args.predict_target_canon_vol and not (epoch==0 and iteration<0):
            data_dict['canon_volume'] = canonical_latent_volume
            with torch.no_grad():
                xy_gen_warp_targ, data_dict['target_delta_xy'] = self.xy_generator_nw(target_warp_embed_dict)
                if self.args.unet_first:
                    _latent_volume = self.volume_process_nw(self.local_encoder_nw(data_dict['target_img'] * data_dict['target_mask']).view(b, c, d, s, s))
                    _latent_volume = self.grid_sample(self.grid_sample(_latent_volume, data_dict['target_inv_rotation_warp']), xy_gen_warp_targ)
                    _canonical_latent_volume = self.volume_source_nw(_latent_volume, embed_dict)
                else:
                    _latent_volume = self.volume_source_nw(self.local_encoder_nw(data_dict['target_img'] * data_dict['target_mask']).view(b, c, d, s, s))
                    _latent_volume = self.grid_sample(self.grid_sample(_latent_volume, data_dict['target_inv_rotation_warp']), xy_gen_warp_targ)
                    _canonical_latent_volume = self.volume_process_nw(_latent_volume, embed_dict)
                data_dict['canon_volume_from_target'] = _canonical_latent_volume
            



        # Warp to target pose
        aligned_target_volume = self.grid_sample(
            self.grid_sample(canonical_latent_volume, data_dict['target_uv_warp_resize']),
            data_dict['target_rotation_warp'])
        
        # Features 3D prepocess
        if self.args.pred_volume_num_blocks > 0:
            aligned_target_volume = self.volume_pred_nw(aligned_target_volume)

        # Get final features before decoder
        if self.args.use_back:
            aligned_target_volume = aligned_target_volume.view(b, c * d, s, s)
            aligned_target_volume = self.backgroung_adding_nw(
                torch.cat((aligned_target_volume, target_latent_feats_back), dim=1))
        else:
            if self.args.volume_rendering:
                aligned_target_volume, data_dict['pred_tar_img_vol'], data_dict['pred_tar_depth_vol'] = self.volume_renderer_nw(aligned_target_volume)
            else:
                aligned_target_volume = aligned_target_volume.view(b, c * d, s, s)

        # Decoder
        data_dict['pred_target_img'], _, _, _ = self.decoder_nw(data_dict, target_warp_embed_dict, aligned_target_volume, False, iteration=iteration) ## _ reserved for some features for stage 2 of smt else

        # If we try to get the same emotion after first warping
        if self.args.match_neutral:
            with contextlib.nullcontext() if (epoch==0 and iteration<200) else torch.no_grad():
                canonical_latent_volume_n = canonical_latent_volume.clone()
                if self.args.use_back:
                    canonical_latent_volume_n = canonical_latent_volume_n.view(b, c * d, s, s)
                    canonical_latent_volume_n = self.backgroung_adding_nw(
                        torch.cat((canonical_latent_volume_n, target_latent_feats_back), dim=1))
                else:
                    if self.args.volume_rendering:
                        canonical_latent_volume_n, data_dict['pred_tar_img_vol'], data_dict['pred_tar_depth_vol'] = self.volume_renderer_nw(canonical_latent_volume_n)
                    else:
                        canonical_latent_volume_n = canonical_latent_volume_n.view(b, c * d, s, s)

                data_dict['pred_neutral_img'], _, _, _ = self.decoder_nw(data_dict, target_warp_embed_dict, canonical_latent_volume_n, False, iteration=iteration)

                s_a = data_dict['pred_neutral_img'].shape[-1]//4

                data_dict['pred_neutral_img_aligned']  = data_dict['pred_neutral_img'][:, :, s_a:3*s_a, s_a:3*s_a]

                data_dict['pred_neutral_expr_vertor'] = self.expression_embedder_nw.net_face(data_dict['pred_neutral_img_aligned'])[0]





        # Get images from flipped texture if needed. Default: False
        if self.training and self.pred_flip:
            data_dict['pred_target_img_flip'] = None

        # Decode into image
        if not self.args.use_back:
            data_dict['target_img'] = data_dict['target_img'] * data_dict['target_mask'].detach()

            if self.args.green:
                green = torch.ones_like(data_dict['target_img'])*(1-data_dict['target_mask'].detach())
                green[:, 0, :, :] = 0
                green[:, 2, :, :] = 0
                data_dict['target_img'] += green

        if self.pred_mixing or not self.training:
            if self.args.use_mix_losses and self.training:

                #########################
                ### Mixing prediction ###
                #########################

                # Predict another warping - from combination of idt vector and rolled expression vector
                mixing_uv_warp_resize, data_dict['target_delta_uv'] = self.uv_generator_nw(mixing_warp_embed_dict)
                if self.resize_warp:
                    mixing_uv_warp_resize = self.resize_warp_func(mixing_uv_warp_resize)

                # Warp canonical target volume to desired emotions
                aligned_mixing_feat = self.grid_sample(canonical_latent_volume, mixing_uv_warp_resize)

                # Getting mixing theta - scales from source all other from target. Sometimes random theta to remove angle from vector
                mixing_theta, self.thetas_pool = get_mixing_theta(self.args, data_dict['source_theta'], data_dict['target_theta'], self.thetas_pool, self.args.random_theta)

                # Warp loatent volume again but by rotation
                mixing_align_warp = self.identity_grid_3d.repeat_interleave(b, dim=0)
                mixing_align_warp = mixing_align_warp.bmm(mixing_theta.transpose(1, 2)).view(b,*mixing_uv_warp_resize.shape[1:4], 3)
                mixing_align_warp_resize = self.resize_warp_func(mixing_align_warp) if self.resize_warp else mixing_align_warp
                aligned_mixing_feat = self.grid_sample(aligned_mixing_feat, mixing_align_warp_resize)


                if self.args.pred_volume_num_blocks > 0:
                    aligned_mixing_feat = self.volume_pred_nw(aligned_mixing_feat)

                # Get final features before decoder
                if self.args.use_back:
                    aligned_mixing_feat = aligned_mixing_feat.view(b, c * d, s, s)
                    aligned_mixing_feat = self.backgroung_adding_nw(
                        torch.cat((aligned_mixing_feat, target_latent_feats_back.detach()), dim=1))
                else:
                    if self.args.volume_rendering:
                        aligned_mixing_feat, data_dict['pred_mixing_img_vol'], data_dict['pred_mixing_depth_vol'] = self.volume_renderer_nw(aligned_mixing_feat)
                    else:
                        aligned_mixing_feat = aligned_mixing_feat.view(b, c * d, s, s)

                # Just to be sure 
                self.decoder_nw.train()



                # Decode images
                data_dict['pred_mixing_img'], pred_mixing_seg, _, _ = self.decoder_nw(data_dict, mixing_warp_embed_dict,
                                                                             aligned_mixing_feat, False, iteration=iteration)

                # Finding mixing mask
                mask_mix_ = self.get_mask.forward(data_dict['pred_mixing_img'])
                data_dict['pred_mixing_masked_img'] = data_dict['pred_mixing_img'] * mask_mix_
                data_dict['pred_mixing_mask'] = mask_mix_


                #########################################
                ### Expression contrastive prediction ###
                #########################################

                ## Getting mixing_img_align for resnet18_fv_mix
            
                data_dict_exp = {k: torch.clone(v) if type(v) is torch.Tensor else v for k, v in data_dict.items()}
                with torch.no_grad():
                    data_dict_exp['target_theta'] = self.head_pose_regressor.forward(data_dict['pred_mixing_img'])
                    data_dict_exp['source_theta'] = self.head_pose_regressor.forward(data_dict['target_img'])
                
                data_dict_exp['rolled_mix'] = data_dict['pred_mixing_img']
                data_dict_exp['source_img'] = data_dict['pred_target_img']
                data_dict_exp['source_mask'] = data_dict['target_mask']
                data_dict_exp['target_img'] = data_dict['pred_mixing_img']
                data_dict_exp['target_mask'] = data_dict['pred_mixing_mask']
                data_dict_exp = self.expression_embedder_nw(data_dict_exp, self.args.estimate_head_pose_from_keypoints,
                                                         self.use_masked_aug, use_aug=False)
                data_dict['mixing_img_align'] = data_dict_exp['target_img_align']


                ## Getting mixing_cycle_exp and pred_cycle_exp for contrastive (expression vector of mixing and prediction images)

                data_dict_exp = {k: torch.clone(v) if type(v) is torch.Tensor else v for k, v in data_dict.items()}
                with torch.no_grad():
                    data_dict_exp['target_theta'] = self.head_pose_regressor.forward(data_dict['pred_mixing_img'].roll(-1, dims=0))
                    data_dict_exp['source_theta'] = self.head_pose_regressor.forward(data_dict['target_img'])
                
                data_dict_exp['rolled_mix'] = data_dict['pred_mixing_img'].roll(-1, dims=0)
                data_dict_exp['source_img'] = data_dict['pred_target_img']
                data_dict_exp['source_mask'] = data_dict['target_mask']
                data_dict_exp['target_img'] = data_dict['pred_mixing_img'].roll(-1, dims=0)
                data_dict_exp['target_mask'] = data_dict['pred_mixing_mask'].roll(-1, dims=0)
                data_dict_exp = self.expression_embedder_nw(data_dict_exp, self.args.estimate_head_pose_from_keypoints,
                                                         self.use_masked_aug, use_aug=False)

                _, target_warp_cycle_exp, _, origin_cycle_exp = self.predict_embed(data_dict_exp)
                data_dict['rolled_mix_align'] = data_dict_exp['target_img_align']
                data_dict['rolled_mix'] = data_dict_exp['rolled_mix']


                data_dict['mixing_cycle_exp'] = data_dict_exp['target_pose_embed']
                data_dict['pred_cycle_exp'] = data_dict_exp['source_pose_embed']




                # Predict cycle (canonical volume warp with expression from mixing - result should match target and prediction 
                # as volume the same and espression on mixing should be the same)
                # default: False

                if self.pred_cycle: 
                    target_uv_warp, data_dict['target_delta_uv'] = self.uv_generator_nw(target_warp_cycle_exp)
                    target_uv_warp_resize = target_uv_warp
                    if self.resize_warp:
                        target_uv_warp_resize = self.resize_warp_func(target_uv_warp_resize)
                
                    pred_cycle_expression_vol = \
                        self.grid_sample(self.grid_sample(canonical_latent_volume, target_uv_warp_resize),
                                         data_dict['target_rotation_warp'])
                
                    if self.args.pred_volume_num_blocks > 0:
                        target_latent_feats = self.volume_pred_nw(target_latent_feats)

                
                    if self.args.use_back:
                        target_latent_feats = pred_cycle_expression_vol.view(b, c * d, s, s)
                      #  target_latent_feats = target_latent_feats + target_latent_feats_back.detach()
                        target_latent_feats = self.backgroung_adding_nw(torch.cat((target_latent_feats, target_latent_feats_back.detach()), dim=1))
                    else:
                        if self.args.volume_rendering:
                            target_latent_feats, data_dict['pred_mixing_img_vol'], data_dict['pred_mixing_depth_vol'] = self.volume_renderer_nw(pred_cycle_expression_vol)
                        else:
                            target_latent_feats = pred_cycle_expression_vol.view(b, c * d, s, s)


                    expression_cycle, _, _, _ = self.decoder_nw(data_dict_exp, origin_cycle_exp, target_latent_feats, False, iteration=iteration)
                    data_dict['cycle_mix_pred'] = expression_cycle[:b]
            else:
                with torch.no_grad(): # Mixing on test
                    mixing_uv_warp, data_dict['target_delta_uv'] = self.uv_generator_nw(mixing_warp_embed_dict)
                    mixing_uv_warp_resize = mixing_uv_warp
                    if self.resize_warp:
                        mixing_uv_warp_resize = self.resize_warp_func(mixing_uv_warp_resize)

                    aligned_mixing_feat = self.grid_sample(canonical_latent_volume, mixing_uv_warp_resize)

                    mixing_theta, self.thetas_pool = get_mixing_theta(self.args, data_dict['source_theta'], data_dict['target_theta'], self.thetas_pool, self.args.random_theta)
                    mixing_align_warp = self.identity_grid_3d.repeat_interleave(b, dim=0)
                    mixing_align_warp = mixing_align_warp.bmm(mixing_theta.transpose(1, 2)).view(b,*mixing_uv_warp.shape[1:4],3)
                    if self.resize_warp:
                        mixing_align_warp_resize = self.resize_warp_func(mixing_align_warp)
                    else:
                        mixing_align_warp_resize = mixing_align_warp

                    aligned_mixing_feat = self.grid_sample(aligned_mixing_feat, mixing_align_warp_resize)


                    if self.args.pred_volume_num_blocks > 0:
                        aligned_mixing_feat = self.volume_pred_nw(aligned_mixing_feat)

                    if self.args.use_back:
                        target_latent_feats = aligned_mixing_feat.view(b, c * d, s, s)
                        target_latent_feats = self.backgroung_adding_nw(torch.cat((target_latent_feats, target_latent_feats_back.detach()), dim=1))
                    else:
                        if self.args.volume_rendering:
                            aligned_mixing_feat, data_dict['pred_mixing_img_vol'], data_dict[
                                'pred_mixing_depth_vol'] = self.volume_renderer_nw(aligned_mixing_feat)
                        else:
                            aligned_mixing_feat = aligned_mixing_feat.view(b, c * d, s, s)

                    self.decoder_nw.eval()

                    if  self.args.use_back:
                        aligned_mixing_feat = self.backgroung_adding_nw(
                            torch.cat((aligned_mixing_feat, target_latent_feats_back.detach()), dim=1))

                    data_dict['pred_mixing_img'], pred_mixing_seg, _, _ = self.decoder_nw(data_dict, mixing_warp_embed_dict,
                                                                                 aligned_mixing_feat, False, iteration=iteration)
                    self.decoder_nw.train()
                    data_dict['pred_mixing_img'] = data_dict['pred_mixing_img'].detach()

        return data_dict

    
    def predict_embed(self, data_dict):
        n = self.num_source_frames
        b = data_dict['source_img'].shape[0] // n
        t = data_dict['target_img'].shape[0] // b

        # with amp.autocast(enabled=self.autocast):
            # Unsqueeze pose embeds for warping gen
        warp_source_embed = self.pose_unsqueeze_nw(data_dict['source_pose_embed']).view(b * n, -1, self.embed_size,
                                                                                        self.embed_size)
        warp_target_embed = self.pose_unsqueeze_nw(data_dict['target_pose_embed']).view(b * t, -1, self.embed_size,
                                                                                        self.embed_size)

        if self.pred_mixing:
            if self.args.detach_warp_mixing_embed:
                warp_mixing_embed = warp_target_embed.detach()
            else:
                warp_mixing_embed = warp_target_embed.clone()
            warp_mixing_embed = warp_mixing_embed.view(b, t, -1, self.embed_size, self.embed_size).roll(1, dims=0)
            rolled_t_emb = data_dict['target_pose_embed'].clone().roll(1, dims=0)
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

        embd = [data_dict['source_pose_embed'], data_dict['target_pose_embed'], rolled_t_emb]
        # Predict warp embeds
        data_dict['idt_embed'] = data_dict['idt_embed']
        for k, (pose_embed, m) in enumerate(zip(pose_embeds, num_frames)):


            if self.args.cat_em:
                warp_embed_orig = self.warp_embed_head_orig_nw(torch.cat([pose_embed, data_dict['idt_embed'].repeat_interleave(m, dim=0)], dim=1))
                warp_embed_orig_d = self.warp_embed_head_orig_nw(torch.cat([pose_embed.detach(), data_dict['idt_embed'].repeat_interleave(m, dim=0)], dim=1))
            else:
                warp_embed_orig = self.warp_embed_head_orig_nw((pose_embed + data_dict['idt_embed'].repeat_interleave(m, dim=0)) * 0.5)
                warp_embed_orig_d = self.warp_embed_head_orig_nw((pose_embed.detach() + data_dict['idt_embed'].repeat_interleave(m, dim=0)) * 0.5)

            c = warp_embed_orig.shape[1]
            warp_embed_dicts[k]['orig'] = warp_embed_orig.view(b * m, c, self.embed_size ** 2)
            warp_embed_dicts[k]['orig_d'] = warp_embed_orig_d.view(b * m, c, self.embed_size ** 2)
            # warp_embed_dicts[k]['ada_v'] = pose_embed.view(b * m, c, self.embed_size ** 2)
            warp_embed_dicts[k]['ada_v'] = embd[k]
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

    def calc_train_losses(self, data_dict: dict, mode: str = 'gen', epoch=0, ffhq_per_b=0, iteration=0):
        return calc_train_losses(self, data_dict=data_dict, mode=mode, epoch=epoch, ffhq_per_b=ffhq_per_b, iteration=iteration)

    def calc_test_losses(self, data_dict: dict, iteration=0):
        return calc_test_losses(self, data_dict, iteration=iteration)

    def prepare_input_data(self, data_dict):
        return prepare_input_data(self, data_dict)

    #######################################################################
    ####################### Forward everithing ############################
    #######################################################################
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

        s = self.args.image_additional_size if self.args.image_additional_size is not None else data_dict['target_img'].shape[-1]
        
        resize = lambda img: F.interpolate(img, mode='bilinear', size=(s, s), align_corners=False)

        if mode == 'gen':
            # Prepare data 
            data_dict = self.prepare_input_data(data_dict)

            ## Run main model
            data_dict = self.G_forward(data_dict, visualize, iteration=iteration, epoch=epoch)


            if phase == 'train':
                # Put both discriminator to eval mode as we use it now as loss for G
                self.discriminator_ds.eval()
                for p in self.discriminator_ds.parameters():
                    p.requires_grad = False

                if self.args.use_mix_dis:
                    self.discriminator2_ds.eval()
                    for p in self.discriminator2_ds.parameters():
                        p.requires_grad = False

                # Without grad as it is ground truth
                with torch.no_grad():
                    _, data_dict['real_feats_gen'] = self.discriminator_ds(resize(data_dict['target_img']))

                    if self.args.use_mix_dis:
                        _, data_dict['real_feats_gen_2'] = self.discriminator2_ds(data_dict['pred_target_img'].clone().detach())

                # With grad as it is predict
                data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator_ds(
                    resize(data_dict['pred_target_img']))

                if self.args.use_mix_dis:
                    data_dict['fake_score_gen_mix'], _ = self.discriminator2_ds(
                        resize(data_dict['pred_mixing_img']))

                # Calculate all losses 
                loss, losses_dict = self.calc_train_losses(data_dict=data_dict, mode='gen', epoch=epoch,
                                                           ffhq_per_b=ffhq_per_b, iteration=iteration)
                
                if self.use_stylegan_d: # default: False. But let's save it for future. 
                    self.stylegan_discriminator_ds.eval()
                    for p in self.stylegan_discriminator_ds.parameters():
                        p.requires_grad = False

                    data_dict['fake_style_score_gen'] = self.stylegan_discriminator_ds(
                        (data_dict['pred_target_img'] - 0.5) * 2)

                    losses_dict["g_style"] = self.weights['stylegan_weight'] * g_nonsaturating_loss(
                        data_dict['fake_style_score_gen'])

                    if self.args.use_mix_losses and epoch >= self.args.mix_losses_start:
                        data_dict['fake_style_score_gen_mix'] = self.stylegan_discriminator_ds(
                            (data_dict['pred_mixing_img'] - 0.5) * 2)

                        losses_dict["g_style"] += self.weights['stylegan_weight'] * g_nonsaturating_loss(
                            data_dict['fake_style_score_gen_mix'])

            elif phase == 'test':
                loss = None
                losses_dict, expl_var, expl_var_test  = self.calc_test_losses(data_dict, iteration=iteration)

                if expl_var is not None:
                    data_dict['expl_var'] = expl_var
                    data_dict['expl_var_test'] = expl_var_test
            
            else:
                raise ValueError(f"Wrong phase name: {phase}")

        elif mode == 'dis':

            if self.args.detach_dis_inputs:
                data_dict['target_img'] = torch.tensor(data_dict['target_img'].detach().clone().data, requires_grad=True)
                # data_dict['target_img'] = torch.tensor(data_dict['target_img'], requires_grad=False)
                data_dict['pred_target_img'] = torch.tensor(data_dict['pred_target_img'].detach().clone().data, requires_grad=True)


            # Backward through dis
            self.discriminator_ds.train()
            for p in self.discriminator_ds.parameters():
                p.requires_grad = True

            if self.args.use_mix_dis:
                self.discriminator2_ds.train()
                for p in self.discriminator2_ds.parameters():
                    p.requires_grad = True

            data_dict['real_score_dis'], _ = self.discriminator_ds(resize(data_dict['target_img']))
            data_dict['fake_score_dis'], _ = self.discriminator_ds(resize(data_dict['pred_target_img'].detach()))

            if self.args.use_mix_dis:
                data_dict['real_score_dis_mix'], _ = self.discriminator2_ds(resize(data_dict['pred_target_img'].clone().detach()))
                data_dict['fake_score_dis_mix'], _ = self.discriminator2_ds(resize(data_dict['pred_mixing_img'].clone().detach()))
            
            # if self.args.use_exp_v_dis:
            #     data_dict['real_score_dis'], _ = self.discriminator_v_ds(torch.cat([data_dict['pred_target_img'].detach(), data_dict['target_pose_embed'].detach()], dim=1))
            #     data_dict['fake_score_dis'], _ = self.discriminator_v_ds(torch.cat([data_dict['mixing_cycle_exp'].detach(), data_dict['target_pose_embed'].detach()], dim=1))


            loss, losses_dict = self.calc_train_losses(data_dict=data_dict, mode='dis', ffhq_per_b=ffhq_per_b, epoch=epoch)

        elif mode == 'dis_stylegan': # default: False. But let's save it for future.
            losses_dict = {}
            self.stylegan_discriminator_ds.train()
            for p in self.stylegan_discriminator_ds.parameters():
                p.requires_grad = True

            d_regularize = iteration % self.args.d_reg_every == 0

            if d_regularize:
                data_dict['target_img'].requires_grad_()
                data_dict['target_img'].retain_grad()

            fake_pred = self.stylegan_discriminator_ds((data_dict['pred_target_img'].detach() - 0.5) * 2)
            real_pred = self.stylegan_discriminator_ds((data_dict['target_img'] - 0.5) * 2)
            losses_dict["d_style"] = d_logistic_loss(real_pred, fake_pred)

            if self.args.use_mix_losses and epoch >= self.args.mix_losses_start:
                fake_pred_mix = self.stylegan_discriminator_ds((data_dict['pred_mixing_img'].detach() - 0.5) * 2)
                fake_loss_mix = F.softplus(fake_pred_mix)
                losses_dict["d_style"] += fake_loss_mix.mean()

            if d_regularize:
                r1_penalty = _calc_r1_penalty(data_dict['target_img'],
                                              real_pred,
                                              scale_number='all',
                                              )
                data_dict['target_img'].requires_grad_(False)
                losses_dict["r1"] = r1_penalty * self.args.d_reg_every * self.args.r1

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



    def visualize_data(self, data_dict):
        return visualize_data(self, data_dict)

    def get_visuals(self, data_dict):
        return get_visuals(self, data_dict)

    @staticmethod
    def draw_stickman(args, poses):
        return draw_stickman(args, poses)

    def gen_parameters(self):

        params = itertools.chain(*([getattr(self, net).parameters() for net in self.opt_net_names]), [getattr(self, tensor) for tensor in self.opt_tensor_names])

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

        if self.args.use_mix_dis:
            opt_dis = opts[self.args.dis_opt_type](
                itertools.chain(self.discriminator_ds.parameters(), self.discriminator2_ds.parameters()),
                self.args.dis_lr,
                self.args.dis_beta1,
                self.args.dis_beta2,
            )
        else:
            opt_dis = opts[self.args.dis_opt_type](
                self.discriminator_ds.parameters(),
                self.args.dis_lr,
                self.args.dis_beta1,
                self.args.dis_beta2,
            )

        if self.use_stylegan_d:
            d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)
            opt_dis_style = opts['adam'](
                self.stylegan_discriminator_ds.parameters(),
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