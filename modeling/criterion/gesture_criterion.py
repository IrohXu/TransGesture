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
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.exist_weight = exist_weight

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
                
        heatmap_loss = (
            self.heatmap_loss(pred_heatmap.squeeze(1), gt_heatmap) * self.heatmap_weight
        )
        # heatmap_loss = torch.mean(heatmap_loss, dim=(-2, -1))
        # heatmap_loss = torch.sum(heatmap_loss.reshape(-1) * gt_inout.reshape(-1))

        # if heatmap_loss > 1e-7:
        #     heatmap_loss = heatmap_loss / torch.sum(gt_inout)
        #     loss_dict["regression loss"] = heatmap_loss
        # else:
        #     loss_dict["regression loss"] = heatmap_loss * 0
        
        loss_dict["regression loss"] = heatmap_loss
        
        exist_loss = (
            self.exist_loss(pred_exist.reshape(-1), gt_exist.reshape(-1))
            * self.exist_weight
        )
        loss_dict["classification loss"] = exist_loss

        return loss_dict
