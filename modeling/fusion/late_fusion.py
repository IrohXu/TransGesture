import torch
from torch import nn
from detectron2.utils.registry import Registry
from typing import Literal, List, Dict, Optional, OrderedDict
from timm.models.resnetv2 import Bottleneck

FUSION_REGISTRY = Registry("FUSION_REGISTRY")
FUSION_REGISTRY.__doc__ = "Registry for fusion module"

from .cross_attention import CrossAttentionBlock, JointCrossAttentionBlock


@FUSION_REGISTRY.register()
class BaseFusion(nn.Module):
    def __init__(
        self,
        gesture_dim: int = 256,
        gaze_dim: int = 256,
        dim: int = 256,
        image_size: int = 518,
        patch_size: int = 14,
        num_layers: int = 0
    ) -> None:
        super().__init__()
        self.gesture_dim = gesture_dim
        self.gaze_dim = gaze_dim
        self.dim = dim
        self.mask_size = image_size // patch_size
        
        self.fusion_module = nn.Sequential(
            nn.ConvTranspose2d(gesture_dim + gesture_dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_gesture, x_gaze):
        x_gesture = x_gesture.reshape(x_gesture.shape[0], self.mask_size, self.mask_size, x_gesture.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w  
        x_gaze = x_gaze.reshape(x_gaze.shape[0], self.mask_size, self.mask_size, x_gaze.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w   
        feats = torch.cat([x_gesture, x_gaze], dim=1)
        feats = self.fusion_module(feats)
        return feats
    

@FUSION_REGISTRY.register()
class ConvFusion(nn.Module):
    def __init__(
        self,
        gesture_dim: int = 256,
        gaze_dim: int = 256,
        dim: int = 256,
        image_size: int = 518,
        patch_size: int = 14,
        num_layers: int = 3
    ) -> None:
        super().__init__()
        self.gesture_dim = gesture_dim
        self.gaze_dim = gaze_dim
        self.dim = dim
        self.mask_size = image_size // patch_size
        self.in_channel = gesture_dim + gesture_dim
        
        self.merge = Bottleneck(
            self.in_channel,
            self.in_channel
        )
        
        self.conv = nn.Sequential(*[Bottleneck(
            self.in_channel,
            self.in_channel
        ) for i in range(num_layers-1)])
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x_gesture, x_gaze):
        x_gesture = x_gesture.reshape(x_gesture.shape[0], self.mask_size, self.mask_size, x_gesture.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w  
        x_gaze = x_gaze.reshape(x_gaze.shape[0], self.mask_size, self.mask_size, x_gaze.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w 
        feats = torch.cat([x_gesture, x_gaze], dim=1)
        
        feats = self.merge(feats)
        
        feats = self.conv(feats)
        feats = self.decoder(feats)
        return feats
    

@FUSION_REGISTRY.register()
class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        gesture_dim: int = 256,
        gaze_dim: int = 256,
        dim: int = 256,
        image_size: int = 518,
        patch_size: int = 14,
        num_layers: int = 3
    ) -> None:
        super().__init__()
        self.gesture_dim = gesture_dim
        self.gaze_dim = gaze_dim
        self.dim = dim
        self.mask_size = image_size // patch_size
        
        self.cross_fusion = nn.Sequential(*[CrossAttentionBlock(
            dim,
            num_heads=8,
            mlp_ratio=4,
            drop_path=0.1
        ) for i in range(num_layers-1)])
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x_gesture, x_gaze):
        x_gesture, _ = self.cross_fusion((x_gesture, x_gaze))
        feats_gesture = x_gesture.reshape(x_gesture.shape[0], self.mask_size, self.mask_size, x_gesture.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w  
        feats = self.decoder(feats_gesture)
        return feats


def build_fusion_module(name, *args, **kwargs):
    return FUSION_REGISTRY.get(name)(*args, **kwargs)

    

@FUSION_REGISTRY.register()
class JointCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        gesture_dim: int = 256,
        gaze_dim: int = 256,
        dim: int = 256,
        image_size: int = 518,
        patch_size: int = 14,
        num_layers: int = 3
    ) -> None:
        super().__init__()
        self.gesture_dim = gesture_dim
        self.gaze_dim = gaze_dim
        self.dim = dim
        self.mask_size = image_size // patch_size
        
        self.cross_fusion = nn.Sequential(*[JointCrossAttentionBlock(
            dim,
            num_heads=16,
            mlp_ratio=4,
            drop_path=0.1
        ) for i in range(num_layers-1)])
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x_gesture, x_gaze):
        x_gesture, _ = self.cross_fusion((x_gesture, x_gaze))
        feats_gesture = x_gesture.reshape(x_gesture.shape[0], self.mask_size, self.mask_size, x_gesture.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w  
        feats = self.decoder(feats_gesture)
        return feats


def build_fusion_module(name, *args, **kwargs):
    return FUSION_REGISTRY.get(name)(*args, **kwargs)

