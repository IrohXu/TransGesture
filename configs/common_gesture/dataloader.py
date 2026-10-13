from os import path as osp
from typing import Literal

from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from detectron2.config import instantiate
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from detectron2.utils import comm
import torch

from data import *

DATA_ROOT = "/projects/illinois/eng/cs/jrehg/users/xucao2/neurips25/GestureTarget2"
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


class BalancedBatchSampler(Sampler):

    def __init__(self, dataset, labels=None, seed=0):
        if labels is None:
            raise Exception("You should pass the tensor of labels to the constructor as second argument")
        self.seed = seed
        self.epoch = 0
        self.num_replicas = comm.get_world_size()
        self.rank = comm.get_rank()
        self.class_indices = [
            torch.nonzero(labels == c, as_tuple=True)[0] for c in torch.unique(labels)
        ]
        self.balanced_max = max(len(idx) for idx in self.class_indices)
        self.total_size = self.balanced_max * len(self.class_indices)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        per_class = []
        for idx in self.class_indices:
            idx = idx[torch.randperm(len(idx), generator=g)]
            extra = self.balanced_max - len(idx)
            if extra > 0:
                idx = torch.cat([idx, idx[torch.randint(len(idx), (extra,), generator=g)]])
            per_class.append(idx)
        indices = torch.stack(per_class, dim=1).reshape(-1)
        indices = indices[self.rank : self.num_samples * self.num_replicas : self.num_replicas]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


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
        sampler=BalancedBatchSampler(dataset, labels=dataset.labels) if is_train else None,
        shuffle=False,
        drop_last=drop_last if is_train else False,
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
