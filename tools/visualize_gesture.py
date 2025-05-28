import sys
from os import path as osp
import argparse
import warnings
import torch
import numpy as np
from PIL import Image
from detectron2.config import instantiate, LazyConfig
import os

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *

import cv2

warnings.simplefilter(action="ignore", category=FutureWarning)



def do_test(cfg, model, visualization_dir):
    val_loader = instantiate(cfg.dataloader.val)

    model.train(False)
    AUC = []
    min_dist = []
    avg_dist = []
    with torch.no_grad():
        for data in val_loader:
            image_path = data["image_path"]
            val_gesture_heatmap_pred, _ = model(data)
                                                
            val_gesture_heatmap_pred = (
                val_gesture_heatmap_pred.squeeze(1).cpu().detach().numpy()
            )
            
            input_images = (
                data["images"].squeeze(1).cpu().detach().numpy()
            )
            
            val_subject_images = (
                data["subject_channels"].squeeze(1).cpu().detach().numpy()
            )
            
            # go through each data point and record AUC, min dist, avg dist
            for b_i in range(len(val_gesture_heatmap_pred)):

                visualization_save_path = osp.join(
                    visualization_dir,
                    image_path[b_i]
                )
                subject_bbox = [int(data["subject_bbox"][0][b_i].cpu().numpy()), int(data["subject_bbox"][1][b_i].cpu().numpy()), int(data["subject_bbox"][2][b_i].cpu().numpy()), int(data["subject_bbox"][3][b_i].cpu().numpy())]
                
                
                scaled_heatmap = np.array(
                    Image.fromarray(val_gesture_heatmap_pred[b_i]).resize(
                        tuple(data["imsize"][b_i].cpu().detach().numpy()),
                        resample=Image.BILINEAR,
                    )
                )
                
                scaled_subject_heatmap = np.array(
                    Image.fromarray(val_subject_images[b_i]).resize(
                        tuple(data["imsize"][b_i].cpu().detach().numpy()),
                        resample=Image.BILINEAR,
                    )
                )
                
                image = input_images[b_i]
                image = image.transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image * std + mean
                image = np.clip(image, 0, 1)
                image = image * 255
                image = image.astype("uint8")
                image = cv2.resize(image, tuple(data["imsize"][b_i].cpu().detach().numpy()))
                
                # cv2.imwrite("image.jpg", image)
                
                visualization_pred = image.copy()
                
                cv2.rectangle(visualization_pred, (subject_bbox[0], subject_bbox[1]), (subject_bbox[2], subject_bbox[3]), (0, 255, 0), 2)
                # overlay = image.copy()
                # overlay[scaled_subject_heatmap > 0.5] = [0, 255, 0]
                # alpha = 0.3  # Transparency of red mask
                # visualization_pred[scaled_subject_heatmap > 0.5] = cv2.addWeighted(image[scaled_subject_heatmap > 0.5], 1 - alpha, overlay[scaled_subject_heatmap > 0.5], alpha, 0)
                
                overlay = image.copy()
                overlay[scaled_heatmap >= 0.5] = [0, 0, 255]
                alpha = 0.3  # Transparency of red mask
                if np.sum(scaled_heatmap >= 0.5) > 0:
                    visualization_pred[scaled_heatmap > 0.5] = cv2.addWeighted(image[scaled_heatmap > 0.5], 1 - alpha, overlay[scaled_heatmap > 0.5], alpha, 0)

                cv2.imwrite(visualization_save_path, visualization_pred)
                

def main(args):
    cfg = LazyConfig.load(args.config_file)
    visualization_dir = osp.join(args.output_path, "visualization")
    if not osp.exists(visualization_dir):
        os.makedirs(visualization_dir)
    model: torch.Module = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.model_weights, weights_only=False)["model"])
    model.to(cfg.train.device)
    do_test(cfg, model, visualization_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file")
    parser.add_argument("--output_path", type=str, help="output path")
    parser.add_argument(
        "--model_weights",
        type=str,
        help="model weights",
    )
    parser.add_argument("--use_dark_inference", action="store_true")
    args = parser.parse_args()
    main(args)