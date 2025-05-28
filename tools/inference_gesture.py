import sys
from os import path as osp
import argparse
import warnings
from typing import Tuple
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from detectron2.config import instantiate, LazyConfig
import os

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *

import cv2

warnings.simplefilter(action="ignore", category=FutureWarning)

def to_numpy(tensor: torch.Tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray: np.ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

def get_bbox_channel(
    x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False
):
    head_box = (
        np.array([x_min / width, y_min / height, x_max / width, y_max / height])
        * resolution
    )
    int_head_box = head_box.astype(int)
    int_head_box = np.clip(int_head_box, 0, resolution - 1)
    if int_head_box[0] == int_head_box[2]:
        if int_head_box[0] == 0:
            int_head_box[2] = 1
        elif int_head_box[2] == resolution - 1:
            int_head_box[0] = resolution - 2
        elif abs(head_box[2] - int_head_box[2]) > abs(head_box[0] - int_head_box[0]):
            int_head_box[2] += 1
        else:
            int_head_box[0] -= 1
    if int_head_box[1] == int_head_box[3]:
        if int_head_box[1] == 0:
            int_head_box[3] = 1
        elif int_head_box[3] == resolution - 1:
            int_head_box[1] = resolution - 2
        elif abs(head_box[3] - int_head_box[3]) > abs(head_box[1] - int_head_box[1]):
            int_head_box[3] += 1
        else:
            int_head_box[1] -= 1
    head_box = int_head_box
    if coordconv:
        unit = np.array(range(0, resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit + i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution, resolution), dtype=np.float32)
        head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


def inference_gesture(image_path, subject_bbox, head_bbox, model):
    img = Image.open(image_path)
    img = img.convert("RGB")
    
    model.train(False)
    
    width, height = img.size
    
    x_min, y_min, x_max, y_max = subject_bbox
    
    subject_channel = get_bbox_channel(
        x_min,
        y_min,
        x_max,
        y_max,
        width,
        height,
        resolution=518,
        coordconv=False,
    ).unsqueeze(0).cuda()
    
    head_masks = torch.zeros(
        518, 518
    )
    
    head_region = get_bbox_channel(
        head_bbox[0],
        head_bbox[1],
        head_bbox[2],
        head_bbox[3],
        width,
        height,
        resolution=518,
        coordconv=False,
    )
    head_masks[head_region > 0] = 1.0
    head_masks = head_masks.unsqueeze(0).cuda()
    
    image_transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    
    img = image_transform(img).cuda()
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        gesture_heatmap_pred, _ = model.inference(img, subject_channel, head_masks)
    
        gesture_heatmap_pred = (
            gesture_heatmap_pred.squeeze(1).cpu().detach().numpy()
        )
    
    output = gesture_heatmap_pred[0]
    output = np.clip(output, 0, 1)
    output = output * 255
    output = output.astype("uint8")
    
    output = cv2.resize(output, (width, height))
    
    cv2.imwrite("output.jpg", output)
    

def inference_gesture_with_gaze(image_path, subject_bbox, head_bbox, model):
    img = Image.open(image_path)
    img = img.convert("RGB")
    
    model.train(False)
    
    width, height = img.size
    
    x_min, y_min, x_max, y_max = subject_bbox
    
    subject_channel = get_bbox_channel(
        x_min,
        y_min,
        x_max,
        y_max,
        width,
        height,
        resolution=518,
        coordconv=False,
    ).unsqueeze(0).cuda()
    
    head_masks = torch.zeros(
        518, 518
    )
    
    head_region = get_bbox_channel(
        head_bbox[0],
        head_bbox[1],
        head_bbox[2],
        head_bbox[3],
        width,
        height,
        resolution=518,
        coordconv=False,
    )
    head_masks[head_region > 0] = 1.0
    head_masks = head_masks.unsqueeze(0).cuda()
    
    image_transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    
    img = image_transform(img).cuda()
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        gesture_heatmap_pred, gaze_heatmap_pred, _ = model.inference_with_gaze(img, subject_channel, head_masks)
    
        gesture_heatmap_pred = (
            gesture_heatmap_pred.squeeze(1).cpu().detach().numpy()
        )
        
        gaze_heatmap_pred = (
            gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
        )
    
    output = gesture_heatmap_pred[0]
    output = np.clip(output, 0, 1)
    output = output * 255
    output = output.astype("uint8")
    
    output = cv2.resize(output, (width, height))
    
    cv2.imwrite("output_gesture.jpg", output)
    
    output = gaze_heatmap_pred[0]
    output = np.clip(output, 0, 1)
    output = output * 255
    output = output.astype("uint8")
    
    output = cv2.resize(output, (width, height))
    
    cv2.imwrite("output_gaze.jpg", output)
    
                
def main(args):
    cfg = LazyConfig.load(args.config_file)
    model: torch.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights, weights_only=False)["model"])
    model.to(cfg.train.device)
    inference_gesture(image_path, subject_bbox, head_bboxes, model)


if __name__ == "__main__":
    config_file = "./configs/gesture_jointcrossfusion_vit_large.py"
    model_weights = "./output/gesture_jointcrossfusion_vit_large/model_final.pth"
    cfg = LazyConfig.load(config_file)
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(model_weights, weights_only=False)["model"])
    model.to(cfg.train.device)
    
    image_path = "../GestureTarget2/data/0002200.jpg"
    subject_bbox = [0,58,255,360]
    head_bboxes = [129,143,149,203]
    # inference_gesture(image_path, subject_bbox, head_bboxes, model)
    inference_gesture_with_gaze(image_path, subject_bbox, head_bboxes, model)