# Third party
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# This project
from utils import args as args_utils



class WarpReg(nn.Module):


    
    def __init__(self, args):
        super(WarpReg, self).__init__()
        self.apply_to = ['source_delta_xy', 'target_delta_uv']
        self.eps = args.eps
        self.reg_type = 'l1'
        self.weight = 50.0
        self.min_weight = 0.0

        self.weight_decay = 0.9
        self.decay_schedule = 50
        self.num_iters = 0

    def forward(self, data_dict):
        if self.num_iters == self.decay_schedule:
            self.weight = max(self.weight * self.weight_decay, max(self.min_weight, self.eps))
            self.num_iters = 1

        if self.weight == self.eps:
            return data_dict

        loss = 0

        for tensor_name in self.apply_to:
            if 'l1' in self.reg_type:
                if isinstance(data_dict[tensor_name], list):
                    loss_ = 0
                    for tensor in data_dict[tensor_name]:
                        loss_ += tensor.abs()

                else:
                    loss_ = data_dict[tensor_name].abs()

            elif 'l2' in self.reg_type:
                if isinstance(data_dict[tensor_name], list):
                    loss_ = 0
                    for tensor in data_dict[tensor_name]:
                        loss_ += tensor**2

                else:
                    loss_ = data_dict[tensor_name]**2

            elif 'tv' in self.reg_type:
                if isinstance(data_dict[tensor_name], list):
                    loss_ = 0
                    for tensor in data_dict[tensor_name]:
                        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
                        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
                        loss_ += ((dx**2).mean() + (dy**2).mean()) / 2.0

                else:
                    dx = data_dict[tensor_name][..., :-1, 1:] - data_dict[tensor_name][..., :-1, :-1]
                    dy = data_dict[tensor_name][..., 1:, :-1] - data_dict[tensor_name][..., :-1, :-1]
                    loss_ = (dx**2 + dy**2) / 2.0

            else:
                raise # Unknown reg_type

            loss += loss_.mean()

        out_loss = loss * self.weight

        if self.weight_decay != 1.0:
            self.num_iters += 1

        return out_loss