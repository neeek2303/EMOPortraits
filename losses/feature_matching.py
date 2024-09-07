import torch
from torch import nn
import torch.nn.functional as F

from typing import List



class FeatureMatchingLoss(nn.Module):
    def __init__(self, loss_type = 'l1', ):
        super(FeatureMatchingLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self,
                real_features: List[List[List[torch.Tensor]]],
                fake_features: List[List[List[torch.Tensor]]]
        ) -> torch.Tensor:
        """
        features: a list of features of different inputs (the third layer corresponds to
                  features of a separate input to each of these discriminators)
        """
        loss = 0

        for real_feats_net, fake_feats_net in zip(real_features, fake_features):
            # *_feats_net corresponds to outputs of a separate discriminator
            loss_net = 0

            for real_feats_layer, fake_feats_layer in zip(real_feats_net, fake_feats_net):
                assert len(real_feats_layer) == 1 or len(real_feats_layer) == len(fake_feats_layer), 'Wrong number of real inputs'
                if len(real_feats_layer) == 1:
                    real_feats_layer = [real_feats_layer[0]] * len(fake_feats_layer)

                for real_feats_layer_i, fake_feats_layer_i in zip(real_feats_layer, fake_feats_layer):
                    if self.loss_type == 'l1':
                        loss_net += F.l1_loss(fake_feats_layer_i, real_feats_layer_i)
                    elif self.loss_type == 'l2':
                        loss_net += F.mse_loss(fake_feats_layer_i, real_feats_layer_i)

            loss_net /= len(fake_feats_layer) # normalize by the number of inputs
            loss_net /= len(fake_feats_net) # normalize by the number of layers
            loss += loss_net

        loss /= len(real_features) # normalize by the number of networks

        return loss

#
# class FeatureMatchingLoss(nn.Module):
#     def __init__(self, loss_type = 'l1', ):
#         super(FeatureMatchingLoss, self).__init__()
#         self.loss_type = loss_type
#
#     def forward(self,
#                 real_features: List[List[torch.Tensor]],
#                 fake_features: List[List[torch.Tensor]],
#                 confs = None,
#         ) -> torch.Tensor:
#         """
#         features: a list of features of different inputs (the third layer corresponds to
#                   features of a separate input to each of these discriminators)
#         """
#         if confs is not None:
#             fake_conf = [torch.cat([conf, flipped_conf]) for conf, flipped_conf in
#                          zip(confs[0], confs[1])]
#         else:
#             fake_conf = [None] * len(real_features[0])
#
#         loss = 0
#         penalty = 0
#
#
#         for real_feats_scale, fake_feats_scale in zip(real_features, fake_features):
#             # *_feats_net corresponds to outputs of a separate scale!
#             loss_scale = 0
#
#             for real_feats_layer, fake_feats_layer, fake_conf_k in zip(real_feats_scale, fake_feats_scale, fake_conf):
#                 assert len(real_feats_layer) == 1 or len(real_feats_layer) == len(fake_feats_layer), 'Wrong number of real inputs'
#
#                 dist = fake_feats_layer - real_feats_layer.detach()
#
#                 if fake_conf_k is not None and dist.shape[2:] != fake_conf_k.shape[2:]:
#                     fake_conf_k = F.interpolate(fake_conf_k, size=dist.shape[2:], mode='bicubic',
#                                                 align_corners=True).clamp(0.05, 20.0)
#                 dist = dist.abs()
#                 dist = dist * fake_conf_k
#                 loss_scale += dist.mean()
#                 penalty -= fake_conf_k.log().mean()
#
#             loss_scale /= len(real_feats_scale) # normalize by the number of layers
#             loss += loss_scale
#         loss /= len(real_features) # normalize by the number of scales
#         penalty /= len(real_features)
#         penalty /= len(real_feats_scale)
#
#
#         return loss, penalty