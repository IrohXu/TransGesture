from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit


class GestureCriterion(nn.Module):
    def __init__(
        self,
        heatmap_weight: float = 100,
        exist_weight: float = 100,
        use_focal_loss: bool = False,
        alpha: float = -1,
        gamma: float = 2,
        dice_weight: float = 0,
        mask_loss_positive_only: bool = False,
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.exist_weight = exist_weight
        self.dice_weight = dice_weight
        self.mask_loss_positive_only = mask_loss_positive_only

        self.heatmap_loss = nn.BCELoss() # nn.MSELoss(reduce=False)

        if use_focal_loss:
            self.exist_loss = partial(
                sigmoid_focal_loss_jit, alpha=alpha, gamma=gamma, reduction="mean"
            )
        else:
            self.exist_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_heatmap,
        pred_exist,
        gt_heatmap,
        gt_exist
    ):
        loss_dict = {}
                
        pred_heatmap = F.interpolate(
            pred_heatmap,
            size=tuple(gt_heatmap.shape[-2:]),
            mode="bilinear",
            align_corners=True,
        )
                
        pred_heatmap = pred_heatmap.squeeze(1)
        if self.mask_loss_positive_only:
            keep = gt_exist.reshape(-1) > 0
            pred_mask, gt_mask = pred_heatmap[keep], gt_heatmap[keep]
        else:
            pred_mask, gt_mask = pred_heatmap, gt_heatmap

        if pred_mask.shape[0] > 0:
            heatmap_loss = self.heatmap_loss(pred_mask, gt_mask) * self.heatmap_weight
        else:
            heatmap_loss = pred_heatmap.sum() * 0
        loss_dict["regression loss"] = heatmap_loss

        if self.dice_weight > 0:
            if pred_mask.shape[0] > 0:
                inter = (pred_mask * gt_mask).flatten(1).sum(-1)
                denom = pred_mask.flatten(1).sum(-1) + gt_mask.flatten(1).sum(-1)
                dice_loss = (1 - (2 * inter + 1) / (denom + 1)).mean()
            else:
                dice_loss = pred_heatmap.sum() * 0
            loss_dict["dice loss"] = dice_loss * self.dice_weight
        
        exist_loss = (
            self.exist_loss(pred_exist.reshape(-1), gt_exist.reshape(-1))
            * self.exist_weight
        )
        loss_dict["classification loss"] = exist_loss

        return loss_dict
