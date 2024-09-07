import torch
import torch.nn.functional as F
from torch import nn

from typing import Union



class SegmentationLoss(nn.Module):
    def __init__(self, loss_type = 'bce_with_logits'):
        super(SegmentationLoss, self).__init__()
        if loss_type == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, 
                pred_segs: Union[torch.Tensor, list], 
                target_segs: Union[torch.Tensor, list]) -> torch.Tensor:
        if isinstance(pred_segs, list):
            # Concat alongside the batch axis
            pred_segs = torch.cat(pred_segs)
            target_segs = torch.cat(target_segs)

        if target_segs.shape[2] != pred_segs.shape[2]:
            target_segs = F.interpolate(target_segs, size=pred_segs.shape[2:], mode='bilinear')

        loss = self.criterion(pred_segs, target_segs)

        return loss