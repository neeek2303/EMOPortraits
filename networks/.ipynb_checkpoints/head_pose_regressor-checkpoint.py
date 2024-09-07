import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models

from argparse import ArgumentParser
import math
import numpy as np
import copy
from scipy import linalg
import itertools
import functools
from typing import Iterator
import collections
import face_alignment

import losses
from utils import args as args_utils
from utils import misc, spectral_norm, weight_init, point_transforms



class Model(nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("model")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--block_type', default='resnet18')
        parser.add_argument('--input_size', default=256, type=int)

        # Optimization options
        parser.add_argument('--opt_type', default='adam')
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--beta1', default=0.0, type=float)
        parser.add_argument('--beta2', default=0.999, type=float)

        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--weight_decay_layers', default='conv2d')
        parser.add_argument('--weight_decay_params', default='weight')

        parser.add_argument('--shd_type', default='cosine')
        parser.add_argument('--shd_max_iters', default=2.5e5, type=int)
        parser.add_argument('--shd_lr_min', default=1e-6, type=int)

        return parser_out

    def __init__(self, args, training = True):
        super(Model, self).__init__()
        self.args = args

        self.init_networks(args, training)

        if training:
            self.init_losses(args)

    def init_networks(self, args, training):
        self.regressor = models.resnet18(num_classes=9)

        class FAWrapper(object):
            def __init__(self):
                self.fa = face_alignment.FaceAlignment(3)
                self.fa.cuda()

            def get_landmarks_from_batch(self, x):
                return self.fa.get_landmarks_from_batch(x.float())

        self.fa = FAWrapper()

        aligned_keypoints = torch.from_numpy(np.load(f'{self.args.project_dir}/data/aligned_keypoints_3d.npy'))
        aligned_keypoints = aligned_keypoints / 256
        aligned_keypoints[:, :2] -= 0.5
        aligned_keypoints *= 2 # map to [-1, 1]
        self.register_buffer('aligned_keypoints', aligned_keypoints[None], persistent=False)

        grid_s = torch.linspace(-1, 1, 64)
        grid_z = torch.linspace(-1, 1, 16)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer('identity_grid_3d', torch.stack([u, v, w, e], dim=3).view(1, -1, 4), persistent=False)

    def init_losses(self, args):
        self.mse = nn.MSELoss()

    def calc_train_losses(self, data_dict: dict):
        losses_dict = {}

        for name in ['scale', 'rotation', 'translation']:
            losses_dict[name] = self.mse(data_dict[f'pred_{name}'], data_dict[name])

        loss = 0
        for k, v in losses_dict.items():
            loss += v

        return loss, losses_dict

    def calc_test_losses(self, data_dict: dict):
        losses_dict = {'mse': self.mse(data_dict['pred_theta'], data_dict['theta'])}

        return losses_dict

    def forward(self, 
                data_dict: dict, 
                phase: str = 'test', 
                optimizer_idx: int = 0, 
                visualize: bool = False):
        assert phase in ['train', 'test']
        data_dict['image'] = data_dict['source_img'].cuda()
        data_dict['image'] = data_dict['image'].view(-1, *data_dict['image'].shape[2:])
        if self.args.use_amp and self.args.amp_opt_level == 'O2':
            data_dict['image'] = data_dict['image'].half()

        s = data_dict['image'].shape[2]

        keypoints = self.fa.get_landmarks_from_batch(data_dict['image'] * 255)

        if keypoints is None:
            return torch.empty(1).requires_grad_(), {}, None, data_dict

        has_keypoints = [keypoints_i is not None and len(keypoints_i) for keypoints_i in keypoints]

        if not all(has_keypoints):
            return torch.empty(1).requires_grad_(), {}, None, data_dict

        keypoints = torch.stack([torch.from_numpy(keypoints_i[0]).cuda() for keypoints_i in keypoints])

        keypoints /= s
        keypoints[..., :2] -= 0.5
        keypoints *= 2

        data_dict['keypoints'] = keypoints

        data_dict['theta'], params = point_transforms.estimate_transform_from_keypoints(data_dict['keypoints'], self.aligned_keypoints)
        data_dict['scale'], data_dict['rotation'], data_dict['translation'] = params

        if self.args.use_amp and self.args.amp_opt_level == 'O2':
            data_dict['theta'] = data_dict['theta'].half()
            data_dict['scale'] = data_dict['scale'].half()
            data_dict['rotation'] = data_dict['rotation'].half()
            data_dict['translation'] = data_dict['translation'].half()

        (data_dict['pred_scale'], 
         data_dict['pred_rotation'], 
         data_dict['pred_translation']) = self.regressor(data_dict['image']).split([3, 3, 3], dim=1)

        data_dict['pred_theta'] = point_transforms.get_transform_matrix(data_dict['pred_scale'], data_dict['pred_rotation'], data_dict['pred_translation'])

        if phase == 'train':
            loss, losses_dict = self.calc_train_losses(data_dict)

        elif phase == 'test':
            loss = None
            losses_dict = self.calc_test_losses(data_dict)

        visuals = None
        if visualize:
            data_dict = self.visualize_data(data_dict)
            visuals = self.get_visuals(data_dict)

        return loss, losses_dict, visuals, data_dict

    @torch.no_grad()
    def visualize_data(self, data_dict):
        b = data_dict['image'].shape[0]
        shape = (-1, 16, 64, 64, 3)

        data_dict['stickman'] = misc.draw_stickman(data_dict['keypoints'], self.args.image_size)

        grid = self.identity_grid_3d.repeat_interleave(b, dim=0)


        if self.args.use_amp and self.args.amp_opt_level == 'O2':
            inv_theta = data_dict['theta'].float().inverse().half()
            inv_pred_theta = data_dict['pred_theta'].float().inverse().half()
        else:
            inv_theta = data_dict['theta'].inverse()
            inv_pred_theta = data_dict['pred_theta'].inverse()

        data_dict['c2w_warp'] = grid.bmm(inv_theta[:, :3].transpose(1, 2)).view(*shape).mean(1)[..., :2]
        data_dict['w2c_warp'] = grid.bmm(data_dict['theta'][:, :3].transpose(1, 2)).view(*shape).mean(1)[..., :2]

        data_dict['pred_c2w_warp'] = grid.bmm(inv_pred_theta[:, :3].transpose(1, 2)).view(*shape).mean(1)[..., :2]
        data_dict['pred_w2c_warp'] = grid.bmm(data_dict['pred_theta'][:, :3].transpose(1, 2)).view(*shape).mean(1)[..., :2]

        data_dict['world_img'] = F.grid_sample(data_dict['image'], data_dict['c2w_warp'])
        data_dict['camera_img'] = F.grid_sample(data_dict['world_img'], data_dict['w2c_warp'])

        data_dict['pred_world_img'] = F.grid_sample(data_dict['image'], data_dict['pred_c2w_warp'])
        data_dict['pred_camera_img'] = F.grid_sample(data_dict['pred_world_img'], data_dict['pred_w2c_warp'])

        return data_dict

    @torch.no_grad()
    def get_visuals(self, data_dict):
        b = data_dict['image'].shape[0]

        visuals = []

        uvs_prep = lambda x: (x.permute(0, 3, 1, 2) + 1) / 2
        segs_prep = lambda x: torch.cat([x] * 3, dim=1)
        scores_prep = lambda x: (x + 1) / 2

        visuals_list = [
            ['stickman', None],
            ['image', None],
            ['c2w_warp', uvs_prep],
            ['world_img', None],
            ['w2c_warp', uvs_prep],
            ['camera_img', None],
            ['pred_c2w_warp', uvs_prep],
            ['pred_world_img', None],
            ['pred_w2c_warp', uvs_prep],
            ['pred_camera_img', None]
        ]

        max_h = max_w = 0

        for tensor_name, preprocessing_op in visuals_list:
            visuals += misc.prepare_visual(data_dict, tensor_name, preprocessing_op)

            if len(visuals):
                h, w = visuals[-1].shape[2:]
                max_h = max(h, max_h)
                max_w = max(w, max_w) 

        # Upsample all tensors to maximum size
        for i, tensor in enumerate(visuals):
            visuals[i] = F.interpolate(tensor, size=(max_h, max_w), mode='bicubic', align_corners=False)

        visuals = torch.cat(visuals, 3) # cat w.r.t. width
        visuals = visuals.clamp(0, 1)

        return visuals

    def configure_optimizers(self):
        opts = {
            'adam': lambda param_groups, lr, beta1, beta2: torch.optim.Adam(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2)),
            'adamw': lambda param_groups, lr, beta1, beta2: torch.optim.AdamW(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2))}

        opt = opts[self.args.opt_type](
            self.regressor.parameters(), 
            self.args.lr, 
            self.args.beta1,
            self.args.beta2)

        return [opt]

    def configure_schedulers(self, opts):
        shds = {
            'step': lambda optimizer, lr_max, lr_min, max_iters: torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=max_iters,
                gamma=lr_max / lr_min),
            'cosine': lambda optimizer, lr_max, lr_min, max_iters: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max_iters,
                eta_min=lr_min)}

        shd = shds[self.args.shd_type](
            opts[0],
            self.args.lr,
            self.args.shd_lr_min,
            self.args.shd_max_iters)

        return [shd], [self.args.shd_max_iters]