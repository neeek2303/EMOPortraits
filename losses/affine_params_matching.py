# Third party
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np



class AffineLoss(nn.Module):

    def __init__(self, args):
        super(AffineLoss, self).__init__()
        self.weight = 20.0
        self.image_size = args.image_size

        aligned_keypoints = torch.from_numpy(np.load(f'{args.project_dir}/data/aligned_keypoints_3d.npy')).float()
        aligned_keypoints /= self.image_size
        aligned_keypoints[:, :2] -= 0.5
        aligned_keypoints *= 2
        self.register_buffer('aligned_keypoints', aligned_keypoints[None])

    def forward(self, data_dict):
        # Compose predicted affine transform matrices
        pred_theta = torch.cat([data_dict['pred_source_theta'], data_dict['pred_target_theta']])
        theta = torch.cat([data_dict['source_theta'], data_dict['target_theta']])

        # print(data_dict['pred_source_theta'].shape, data_dict['source_theta'].shape)
        loss = ((pred_theta - theta.detach())**2).mean() * self.weight

        return loss