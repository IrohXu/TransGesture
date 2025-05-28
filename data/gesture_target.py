from os import path as osp
from typing import Callable, Optional
import json

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd

from . import gesture_augmentation
from .masking import MaskGenerator
from . import data_utils as utils


def bbox_to_mask(image_shape, bbox, value=1):
    """
    Convert a bounding box to a binary mask.

    Args:
        image_shape (tuple): (height, width) of the mask.
        bbox (list or tuple): [x_min, y_min, x_max, y_max] in pixel coordinates.
        value (int): Value to fill the mask (default: 255 for white).
    
    Returns:
        mask (np.ndarray): Binary mask with the same height and width as the input image.
    """
    mask = torch.zeros(image_shape)
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Clip to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_shape[1], x_max)
    y_max = min(image_shape[0], y_max)

    mask[y_min:y_max, x_min:x_max] = value
    return mask


class GestureTarget(Dataset):
    def __init__(
        self,
        image_root: str,
        anno_root: str,
        target_root: str,
        transform: Callable,
        input_size: int,
        output_size: int,
        quant_labelmap: bool = True,
        is_train: bool = True,
        *,
        mask_generator: Optional[MaskGenerator] = None,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
    ):
        
        with open(anno_root, "r") as f:
            data = json.load(f)
            
        self.data = data
        self.length = len(data)
        
        self.labels = []
        for item in data:
            if item["gesture_exist"] == 1:
                self.labels.append(1)
            else:
                self.labels.append(0)
        self.labels = torch.tensor(self.labels)
        
        self.data_dir = image_root
        self.target_dir = target_root
        self.transform = transform
        self.is_train = is_train

        self.input_size = input_size
        self.output_size = output_size

        self.draw_labelmap = (
            utils.draw_labelmap if quant_labelmap else utils.draw_labelmap_no_quant
        )

        if self.is_train:
            ## data augmentation
            self.augment = gesture_augmentation.AugmentationList(
                [
                    gesture_augmentation.ColorJitter(color_jitter),
                    gesture_augmentation.BoxJitter(bbox_jitter),
                    gesture_augmentation.RandomCrop(rand_crop),
                    gesture_augmentation.RandomFlip(rand_flip),
                    gesture_augmentation.RandomRotate(rand_rotate),
                    gesture_augmentation.RandomLSJ(rand_lsj),
                ]
            )

            self.mask_generator = mask_generator

    def __getitem__(self, index):
        item = self.data[index]
        path = item["path"]
        x_min = item["subject_bbox"][0]
        y_min = item["subject_bbox"][1]
        x_max = item["subject_bbox"][2]
        y_max = item["subject_bbox"][3]
        target_bboxes = item["target_bboxes"]
        head_bboxes = item["head_bbox"]
        
        gesture_inside = bool(item["gesture_exist"])
        
        img = Image.open(osp.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
            
        target_mask = Image.open(osp.join(self.target_dir, path))
        target_mask = target_mask.convert('L')
            
        head_bboxes_new = []
        for head_bbox in head_bboxes:
            hx_min = head_bbox[0]
            hy_min = head_bbox[1]
            hx_max = head_bbox[2]
            hy_max = head_bbox[3]
            hx_min, hy_min, hx_max, hy_max = map(float, [hx_min, hy_min, hx_max, hy_max])
            if hx_max < hx_min:
                hx_min, hx_max = hx_max, hx_min
            if hy_max < hy_min:
                hy_min, hy_max = hy_max, hy_min
            head_bboxes_new.append([hx_min, hy_min, hx_max, hy_max])
        head_bboxes = head_bboxes_new
        
        target_bboxes_new = []
        for target_bbox in target_bboxes:
            tx_min = target_bbox[0]
            ty_min = target_bbox[1]
            tx_max = target_bbox[2]
            ty_max = target_bbox[3]
            tx_min, ty_min, tx_max, ty_max = map(float, [tx_min, ty_min, tx_max, ty_max])
            if tx_max < tx_min:
                tx_min, tx_max = tx_max, tx_min
            if ty_max < ty_min:
                ty_min, ty_max = ty_max, ty_min
            target_bboxes_new.append([tx_min, ty_min, tx_max, ty_max])
        target_bboxes = target_bboxes_new
        
        if self.is_train:
            img, target_mask, bbox, head_bboxes, target_bboxes, size = self.augment(
                img,
                target_mask,
                (x_min, y_min, x_max, y_max),
                head_bboxes,
                target_bboxes,
                (width, height),
            )
            x_min, y_min, x_max, y_max = bbox
            width, height = size
            
        subject_channel = utils.get_head_box_channel(
            x_min,
            y_min,
            x_max,
            y_max,
            width,
            height,
            resolution=self.input_size,
            coordconv=False,
        ).unsqueeze(0)
        
        head_masks = torch.zeros(
            self.input_size, self.input_size
        )
        for head_bbox in head_bboxes:
            head_region = utils.get_head_box_channel(
                head_bbox[0],
                head_bbox[1],
                head_bbox[2],
                head_bbox[3],
                width,
                height,
                resolution=self.input_size,
                coordconv=False,
            )
            head_masks[head_region > 0] = 1.0
        head_masks = head_masks.unsqueeze(0)
        
        if self.is_train and self.mask_generator is not None:
            image_mask = self.mask_generator(
                x_min / width,
                y_min / height,
                x_max / width,
                y_max / height,
                subject_channel,
            )

        if self.transform is not None:
            img = self.transform(img)
        
        if self.is_train:
            # target_heatmap = torch.zeros(
            #     self.output_size, self.output_size
            # )
            # for target_bbox in target_bboxes:
            #     target_region = utils.get_head_box_channel(
            #         target_bbox[0],
            #         target_bbox[1],
            #         target_bbox[2],
            #         target_bbox[3],
            #         width,
            #         height,
            #         resolution=self.output_size,
            #         coordconv=False,
            #     )
            #     target_heatmap[target_region > 0] = 1.0
            target_heatmap = torch.zeros(
                self.output_size, self.output_size
            )
            target_mask = TF.to_tensor(
                TF.resize(target_mask, (self.output_size, self.output_size))
            )
            target_heatmap[target_mask[0] > 0] = 1.0
        else:
            target_heatmap = torch.zeros(
                width, height
            )
            target_mask = TF.to_tensor(
                TF.resize(target_mask, (width, height))
            )
            target_heatmap[target_mask[0] > 0] = 1.0
            
        imsize = torch.IntTensor([width, height])
        
        if self.is_train:
            out_dict = {
                "images": img,
                "subject_channels": subject_channel,
                "head_masks": head_masks,
                "heatmaps": target_heatmap,
                "gesture_exists": torch.FloatTensor([gesture_inside]),
                "imsize": imsize,
            }
            if self.mask_generator is not None:
                out_dict["image_masks"] = image_mask
            return out_dict
        else:
            return {
                "images": img,
                "subject_channels": subject_channel,
                "head_masks": head_masks,
                "heatmaps": target_heatmap,
                "gesture_exists": torch.FloatTensor([gesture_inside]),
                "imsize": imsize,
                "image_path": path,
                "subject_bbox": item["subject_bbox"]
            }

    def __len__(self):
        return self.length



