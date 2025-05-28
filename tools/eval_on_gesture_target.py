import sys
from os import path as osp
import argparse
import warnings
import torch
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *

import cv2

warnings.simplefilter(action="ignore", category=FutureWarning)


def average_precision_segmentation(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute the Average Precision (area under the precision–recall curve)
    between a predicted mask and a ground‐truth mask.

    Args:
        pred_mask (np.ndarray): H×W array of float scores (e.g. probabilities).
        gt_mask   (np.ndarray): H×W binary array (0 or 1).

    Returns:
        float: Average Precision (AP) score.
    """
    # flatten to 1D
    pred_flat = pred_mask.ravel()
    gt_flat = gt_mask.ravel().astype(int)

    # edge‐case: no positives in GT
    if gt_flat.sum() == 0:
        return 0.0

    return average_precision_score(gt_flat, pred_flat)


def calculate_multiclass_miou(pred_mask, gt_mask, num_classes=2, ignore_index=None):
    """
    Calculate mIoU for multi-class segmentation.

    Args:
        pred_mask (np.ndarray): Predicted class mask of shape (H, W), values in [0, num_classes-1].
        gt_mask (np.ndarray): Ground truth class mask of shape (H, W), values in [0, num_classes-1].
        num_classes (int): Total number of classes.
        ignore_index (int, optional): Class index to ignore (e.g., background).

    Returns:
        float: Mean IoU across all classes (excluding ignored ones).
        dict: Per-class IoU values.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Shape mismatch between prediction and ground truth.")

    ious = []
    iou_dict = {}

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)
        iou_dict[cls] = iou

    miou = np.mean(ious)
    return miou, iou_dict


def print_model_size(model: torch.nn.Module, verbose: bool = False) -> None:
    """
    Prints the total and trainable parameter count of a PyTorch model.

    Args:
        model (torch.nn.Module): your model
        verbose (bool): if True, also prints per‐module parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    if verbose:
        print("\nPer‐module parameter breakdown:")
        for name, module in model.named_modules():
            pm = sum(p.numel() for p in module.parameters(recurse=False))


def do_test(cfg, model):
    val_loader = instantiate(cfg.dataloader.val)

    model.train(False)
    # iou_single = []
    iou = []
    AP_mask = []
    AUC = []
    exist_pred_list = []
    exist_gt_list = []
    min_dist = []
    avg_dist = []
        
    with torch.no_grad():
        for data in val_loader:
            val_gesture_heatmap_pred, exist_pred = model(data)
            ground_truth = data['heatmaps']
                                                
            val_gesture_heatmap_pred = (
                val_gesture_heatmap_pred.squeeze(1).cpu().detach().numpy()
            )
            
            input_images = (
                data["images"].squeeze(1).cpu().detach().numpy()
            )
            
            val_subject_images = (
                data["subject_channels"].squeeze(1).cpu().detach().numpy()
            )
            
            exist_pred = exist_pred.cpu().detach().numpy()
            exist_pred[exist_pred >= 0.5] = 1.0
            exist_pred[exist_pred < 0.5] = 0.0
            exist_gt = data['gesture_exists'].squeeze(1).cpu().detach().numpy()
            
            exist_pred_list += exist_pred.tolist()
            exist_gt_list += exist_gt.tolist()
            
            # go through each data point and record AUC, min dist, avg dist
            for b_i in range(len(val_gesture_heatmap_pred)):
                
                scaled_heatmap = np.array(
                    Image.fromarray(val_gesture_heatmap_pred[b_i]).resize(
                        tuple(data["imsize"][b_i].cpu().detach().numpy()),
                        resample=Image.BILINEAR,
                    )
                )
                
                # scaled_subject_heatmap = np.array(
                #     Image.fromarray(val_subject_images[b_i]).resize(
                #         tuple(data["imsize"][b_i].cpu().detach().numpy()),
                #         resample=Image.BILINEAR,
                #     )
                # )
                
                gt_mask = ground_truth[b_i].cpu().detach().numpy()
                gt_mask = gt_mask.transpose(1, 0)
                                
                if np.sum(gt_mask) > 0:                
                    AP_mask.append(
                        average_precision_segmentation(scaled_heatmap, gt_mask)
                    )
                
                scaled_heatmap[scaled_heatmap > 0.5] = 1
                scaled_heatmap[scaled_heatmap <= 0.5] = 0
                
                # if np.sum(gt_mask) > 0:                
                #     iou.append(calculate_multiclass_miou(scaled_heatmap, gt_mask)[0])
                                
                iou.append(calculate_multiclass_miou(scaled_heatmap, gt_mask, ignore_index=0)[0])
                
                
    print_model_size(model)
    print("| Exist Acc | mIoU | AP_mask |")
    print(
        "|  {:.4f}   |{:.4f}| {:.4f}  |".format(
            accuracy_score(exist_gt_list, exist_pred_list),
            torch.mean(torch.tensor(iou)),
            torch.mean(torch.tensor(AP_mask)), 
        )
    )


def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights, weights_only=False)["model"])
    model.to(cfg.train.device)
    do_test(cfg, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file")
    parser.add_argument(
        "--model_weights",
        type=str,
        help="model weights",
    )
    args = parser.parse_args()
    main(args)