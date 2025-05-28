from os import path as osp
from typing import Literal

from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from detectron2.config import instantiate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import random

from data import *

DATA_ROOT = ""
# DATA_ROOT = "${Root to Datasets}"
if DATA_ROOT == "${Root to Datasets}":
    raise Exception(
        f"""{osp.abspath(__file__)}: Rewrite `DATA_ROOT` with the root to the datasets.
The directory structure should be:
-DATA_ROOT
"""
)

# Basic Config for Video Attention Target dataset and preprocessing
data_info = OmegaConf.create()

data_info.gesture_target = OmegaConf.create()
data_info.gesture_target.train_root = osp.join(DATA_ROOT, "data")
data_info.gesture_target.train_mask_root = osp.join(DATA_ROOT, "target_mask")
data_info.gesture_target.train_anno = osp.join(
    DATA_ROOT, "train.json"
)
data_info.gesture_target.val_root = osp.join(DATA_ROOT, "data")
data_info.gesture_target.val_mask_root = osp.join(DATA_ROOT, "target_mask")
data_info.gesture_target.val_anno = osp.join(
    DATA_ROOT, "test.json"
)

data_info.input_size = 518
data_info.output_size = 128
data_info.quant_labelmap = True
data_info.mean = (0.485, 0.456, 0.406)
data_info.std = (0.229, 0.224, 0.225)
data_info.bbox_jitter = 0.5
data_info.rand_crop = 0.5
data_info.rand_flip = 0.5
data_info.color_jitter = 0.5
data_info.rand_rotate = 0.0
data_info.rand_lsj = 0.0

data_info.mask_size = 24
data_info.mask_scene = False
data_info.mask_head = False
data_info.max_scene_patches_ratio = 0.5
data_info.max_head_patches_ratio = 0.3
data_info.mask_prob = 0.2

data_info.seq_len = 16
data_info.max_len = 32


class BalancedBatchSampler(DistributedSampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)


# Dataloader(gazefollow/video_atention_target, train/val)
def __build_dataloader(
    name: Literal[
        "gesture_target"
    ],
    is_train: bool,
    batch_size: int = 64,
    num_workers: int = 14,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    **kwargs,
):
    assert name in [
        "gesture_target",
    ], f'{name} not in ("gesture_target")'

    for k, v in kwargs.items():
        if k in ["train_root", "train_anno", "val_root", "val_anno"]:
            data_info[name][k] = v
        else:
            data_info[k] = v

    datasets = {
        "gesture_target": GestureTarget,
    }
    dataset = L(datasets[name])(
        image_root=data_info[name]["train_root" if is_train else "val_root"],
        anno_root=data_info[name]["train_anno" if is_train else "val_anno"],
        target_root=data_info[name]["train_mask_root" if is_train else "val_mask_root"],
        transform=get_transform(
            input_resolution=data_info.input_size,
            mean=data_info.mean,
            std=data_info.std,
        ),
        input_size=data_info.input_size,
        output_size=data_info.output_size,
        quant_labelmap=data_info.quant_labelmap,
        is_train=is_train,
        bbox_jitter=data_info.bbox_jitter,
        rand_crop=data_info.rand_crop,
        rand_flip=data_info.rand_flip,
        color_jitter=data_info.color_jitter,
        rand_rotate=data_info.rand_rotate,
        rand_lsj=data_info.rand_lsj,
        mask_generator=(None),
    )
    dataset = instantiate(dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=None,
        # sampler=DistributedSampler(dataset, shuffle=is_train) if distributed else None,
        sampler=BalancedBatchSampler(dataset, labels=dataset.labels),
        drop_last=drop_last,
    )


dataloader = OmegaConf.create()
dataloader.gesture_target = OmegaConf.create()
dataloader.gesture_target.train = L(__build_dataloader)(
    name="gesture_target",
    is_train=True,
)
dataloader.gesture_target.val = L(__build_dataloader)(
    name="gesture_target",
    is_train=False,
)
