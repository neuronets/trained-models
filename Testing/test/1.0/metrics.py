import torch
from torch import nn
from torch.nn import functional as F


class OCCBCELogits(nn.Module):
    def __init__(self):
        super(OCCBCELogits, self).__init__()

    def forward(self, logits, occ):
        pos_weights = (torch.sum(1. - occ.view(-1, occ.size(-1)), dim=0) + 1e-12) 
        pos_weights = pos_weights  / (torch.sum(occ.view(-1, occ.size(-1)), dim=0) + 1e-12)
        loss = F.binary_cross_entropy_with_logits(logits, occ, pos_weight=pos_weights, reduction='none')
        return loss.sum(dim=[-1,-2]).mean()


class SDFL1Loss(nn.Module):
    def __init__(self):
        super(SDFL1Loss, self).__init__()

    def forward(self, logits, sdf):
        loss = F.l1_loss(logits, sdf, reduction='none')
        return loss.sum(dim=[-1,-2]).mean()


def itersection_over_union(pred_bin, gt_bin):
    assert pred_bin.shape == gt_bin.shape

    batch_size = gt_bin.shape[0]
    pred_bin = pred_bin.reshape(batch_size, -1).bool()
    gt_bin = gt_bin.reshape(batch_size, -1).bool()
    
    # Compute IOU
    area_union = torch.logical_or(pred_bin, gt_bin).float().sum(dim=-1)
    area_intersect = torch.logical_and(pred_bin, gt_bin).float().sum(dim=-1)
    iou = (area_intersect / area_union)
    return iou.mean()