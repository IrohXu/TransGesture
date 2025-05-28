import torch
from torch import nn
from typing import Dict, Union
from ..backbone.vit import Block
import torchvision

import math

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)


class GestureFusionMapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        fusion: nn.Module,
        criterion: nn.Module,
        device: Union[torch.device, str] = "cuda",
        freeze_backbone: bool = True,
        freeze_gaze_branch: bool = True,
        gaze_dim: int = 256,
        gesture_dim: int = 256,
        gaze_inout: bool = True,
        gesture_exist: bool = True,
        gaze_num_layers: int = 3,
        gesture_num_layers: int = 1,
        image_size: int = 518,
        patch_size: int = 14,
        output_size: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion
        self.criterion = criterion
        self.device = torch.device(device)
        # self.dim = dim
        self.gaze_dim = gaze_dim
        self.gesture_dim = gesture_dim
        self.linear = nn.Conv2d(self.backbone.embed_dim, self.gaze_dim, 1)
        self.linear_gesture = nn.Conv2d(self.backbone.embed_dim, self.gesture_dim, 1)
        self.subject_token = nn.Embedding(1, self.gesture_dim)
        self.head_token = nn.Embedding(1, self.gaze_dim)
        
        self.projection_gaze2gesture = nn.Linear(self.gaze_dim, self.gesture_dim)
        
        self.mask_size = image_size // patch_size
        self.register_buffer("pos_embed", positionalencoding2d(self.gaze_dim, self.mask_size, self.mask_size).squeeze(dim=0).squeeze(dim=0))
        self.register_buffer("gesture_pos_embed", positionalencoding2d(self.gesture_dim, self.mask_size, self.mask_size).squeeze(dim=0).squeeze(dim=0))
        self.gaze_inout = gaze_inout
        self.gesture_exist = gesture_exist
        self.out_size = (output_size, output_size)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.gaze_dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1) for i in range(gaze_num_layers)]
        )
        
        if freeze_gaze_branch:
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.linear.parameters():
                param.requires_grad = False
            for param in self.head_token.parameters():
                param.requires_grad = False
        
        self.transformer_gesture = nn.Sequential(*[
            Block(
                dim=self.gesture_dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1) for i in range(gesture_num_layers)]
        )
        
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(self.gaze_dim, self.gaze_dim, kernel_size=2, stride=2),
            nn.Conv2d(self.gaze_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        if self.gaze_inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.gaze_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                # nn.Sigmoid()
            )
            self.inout_token = nn.Embedding(1, self.gaze_dim)
        
        # if self.inout:
        #     self.gesture_inout_head = nn.Sequential(
        #         nn.Linear(self.dim, 128),
        #         nn.ReLU(),
        #         nn.Dropout(0.1),
        #         nn.Linear(128, 1),
        #         # nn.Sigmoid()
        #     )
        #     self.gesture_inout_token = nn.Embedding(1, self.dim)
        
        if self.gesture_exist:
            self.gesture_exist_head = nn.Sequential(
                nn.Linear(self.gesture_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                # nn.Sigmoid()
            )
            self.exist_token_gesture = nn.Embedding(1, self.gesture_dim)

    def forward(self, x):
        (
            scenes,
            subject_masks,
            head_masks,
            gt_heatmaps,
            gt_exists,
        ) = self.preprocess_inputs(x)
        # Calculate patch weights based on head position
        
        embedded_subject = torchvision.transforms.functional.resize(subject_masks, (self.mask_size, self.mask_size))
        embedded_subject = torch.where(embedded_subject > 0, 1., 0.)
        
        embedded_head = torchvision.transforms.functional.resize(head_masks, (self.mask_size, self.mask_size))
        embedded_head = torch.where(embedded_head > 0, 1., 0.)
                
        # Get out-dict
        x = self.backbone(
            scenes,
            None,
            None,
        )
        
        # Apply patch weights to get the final feats and attention maps
        feats = x.get("last_feat", None)
        feats_gaze = self.linear(feats)
        feats_gaze = feats_gaze + self.pos_embed
        
        feats_gesture = self.linear_gesture(feats)
        feats_gesture = feats_gesture + self.gesture_pos_embed

        head_map_embeddings = embedded_head * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        subject_map_embeddings = embedded_subject * self.subject_token.weight.unsqueeze(-1).unsqueeze(-1)
        
        feats_gaze = feats_gaze + head_map_embeddings
        feats_gaze = feats_gaze.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"
        
        feats_gesture = feats_gesture + subject_map_embeddings
        feats_gesture = feats_gesture.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"
        
        # if self.inout:
        #     feats_gesture = torch.cat([self.gesture_inout_token.weight.unsqueeze(dim=0).repeat(feats_gesture.shape[0], 1, 1), feats_gesture], dim=1)
        
        if self.gaze_inout:
            feats_gaze = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(feats_gaze.shape[0], 1, 1), feats_gaze], dim=1)
        
        if self.gesture_exist:
            feats_gesture = torch.cat([self.exist_token_gesture.weight.unsqueeze(dim=0).repeat(feats_gesture.shape[0], 1, 1), feats_gesture], dim=1)

        feats_gaze = self.transformer(feats_gaze)
        feats_gesture = self.transformer_gesture(feats_gesture)
        
        # if self.inout:
        #     inout_tokens = feats_gesture[:, 0, :] 
        #     inout_preds = self.gesture_inout_head(inout_tokens).squeeze(dim=-1)
        #     # inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
        #     feats_gesture = feats_gesture[:, 1:, :] # slice off inout tokens from scene tokens
        
        if self.gaze_inout:
            # inout_tokens = feats_gaze[:, 0, :] 
            # inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            feats_gaze = feats_gaze[:, 1:, :] # slice off inout tokens from scene tokens
            
                
        feats_gaze = self.projection_gaze2gesture(feats_gaze)
        
        if self.gesture_exist:
            exist_tokens = feats_gesture[:, 0, :] 
            exist_preds = self.gesture_exist_head(exist_tokens).squeeze(dim=-1)
            feats_gesture = feats_gesture[:, 1:, :]
        
        # feats_gesture = feats_gesture.reshape(feats_gesture.shape[0], self.mask_size, self.mask_size, feats_gesture.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w  
        # feats_gaze = feats_gaze.reshape(feats_gaze.shape[0], self.mask_size, self.mask_size, feats_gaze.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w              
        
        # feats = torch.cat([feats_gaze, feats_gesture], dim=1)
        # feats = self.heatmap_head(feats)
        
        feats_out = self.fusion(feats_gaze, feats_gesture)
        
        # feats = self.heatmap_head(feats).squeeze(dim=1)
        feats_out = torchvision.transforms.functional.resize(feats_out, self.out_size)
        heatmap_preds = feats_out
            
        if self.training:
            return self.criterion(
                heatmap_preds,
                exist_preds,
                gt_heatmaps,
                gt_exists
            )
        # Inference
                
        return heatmap_preds, exist_preds.sigmoid()
        # return heatmap_preds, inout_preds.sigmoid()

    def preprocess_inputs(self, batched_inputs: Dict[str, torch.Tensor]):
        return (
            batched_inputs["images"].to(self.device),
            batched_inputs["subject_channels"].to(self.device),
            batched_inputs["head_masks"].to(self.device),
            batched_inputs["heatmaps"].to(self.device)
            if "heatmaps" in batched_inputs.keys()
            else None,
            batched_inputs["gesture_exists"].to(self.device)
            if "gesture_exists" in batched_inputs.keys()
            else None,
        )
    
    
    def inference(self, scenes, subject_masks, head_masks):
        
        embedded_subject = torchvision.transforms.functional.resize(subject_masks, (self.mask_size, self.mask_size))
        embedded_subject = torch.where(embedded_subject > 0, 1., 0.)
        
        embedded_head = torchvision.transforms.functional.resize(head_masks, (self.mask_size, self.mask_size))
        embedded_head = torch.where(embedded_head > 0, 1., 0.)
                
        # Get out-dict
        x = self.backbone(
            scenes,
            None,
            None,
        )
        
        feats = x.get("last_feat", None)
        feats_gaze = self.linear(feats)
        feats_gaze = feats_gaze + self.pos_embed
        
        feats_gesture = self.linear_gesture(feats)
        feats_gesture = feats_gesture + self.gesture_pos_embed

        head_map_embeddings = embedded_head * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        subject_map_embeddings = embedded_subject * self.subject_token.weight.unsqueeze(-1).unsqueeze(-1)
        
        feats_gaze = feats_gaze + head_map_embeddings
        feats_gaze = feats_gaze.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"
        
        feats_gesture = feats_gesture + subject_map_embeddings
        feats_gesture = feats_gesture.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"
        
        if self.gaze_inout:
            feats_gaze = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(feats_gaze.shape[0], 1, 1), feats_gaze], dim=1)
        
        if self.gesture_exist:
            feats_gesture = torch.cat([self.exist_token_gesture.weight.unsqueeze(dim=0).repeat(feats_gesture.shape[0], 1, 1), feats_gesture], dim=1)

        feats_gaze = self.transformer(feats_gaze)
        feats_gesture = self.transformer_gesture(feats_gesture)
        
        if self.gaze_inout:
            feats_gaze = feats_gaze[:, 1:, :] # slice off inout tokens from scene tokens
                
        feats_gaze = self.projection_gaze2gesture(feats_gaze)
        
        if self.gesture_exist:
            exist_tokens = feats_gesture[:, 0, :] 
            exist_preds = self.gesture_exist_head(exist_tokens).squeeze(dim=-1)
            feats_gesture = feats_gesture[:, 1:, :]
        
        feats_out = self.fusion(feats_gaze, feats_gesture)
        
        feats_out = torchvision.transforms.functional.resize(feats_out, self.out_size)
        heatmap_preds = feats_out
            
        return heatmap_preds, exist_preds.sigmoid()
    
    
    def inference_with_gaze(self, scenes, subject_masks, head_masks):
        
        embedded_subject = torchvision.transforms.functional.resize(subject_masks, (self.mask_size, self.mask_size))
        embedded_subject = torch.where(embedded_subject > 0, 1., 0.)
        
        embedded_head = torchvision.transforms.functional.resize(head_masks, (self.mask_size, self.mask_size))
        embedded_head = torch.where(embedded_head > 0, 1., 0.)
                
        # Get out-dict
        x = self.backbone(
            scenes,
            None,
            None,
        )
        
        feats = x.get("last_feat", None)
        feats_gaze = self.linear(feats)
        feats_gaze = feats_gaze + self.pos_embed
        
        feats_gesture = self.linear_gesture(feats)
        feats_gesture = feats_gesture + self.gesture_pos_embed

        head_map_embeddings = embedded_head * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        subject_map_embeddings = embedded_subject * self.subject_token.weight.unsqueeze(-1).unsqueeze(-1)
        
        feats_gaze = feats_gaze + head_map_embeddings
        feats_gaze = feats_gaze.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"
        
        feats_gesture = feats_gesture + subject_map_embeddings
        feats_gesture = feats_gesture.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"
        
        if self.gaze_inout:
            feats_gaze = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(feats_gaze.shape[0], 1, 1), feats_gaze], dim=1)
        
        if self.gesture_exist:
            feats_gesture = torch.cat([self.exist_token_gesture.weight.unsqueeze(dim=0).repeat(feats_gesture.shape[0], 1, 1), feats_gesture], dim=1)

        feats_gaze = self.transformer(feats_gaze)
        feats_gesture = self.transformer_gesture(feats_gesture)
        
        if self.gaze_inout:
            feats_gaze = feats_gaze[:, 1:, :] # slice off inout tokens from scene tokens
        
        feats_gaze_out = feats_gaze.reshape(feats_gaze.shape[0], self.mask_size, self.mask_size, feats_gaze.shape[2]).permute(0, 3, 1, 2)
        self.heatmap_head(feats_gaze_out)
        feats_gaze_out = torchvision.transforms.functional.resize(feats_gaze_out, (64, 64))
                        
        feats_gaze = self.projection_gaze2gesture(feats_gaze)
        
        if self.gesture_exist:
            exist_tokens = feats_gesture[:, 0, :] 
            exist_preds = self.gesture_exist_head(exist_tokens).squeeze(dim=-1)
            feats_gesture = feats_gesture[:, 1:, :]
        
        feats_out = self.fusion(feats_gaze, feats_gesture)
        
        feats_out = torchvision.transforms.functional.resize(feats_out, self.out_size)
        heatmap_preds = feats_out
            
        return heatmap_preds, feats_gaze_out, exist_preds.sigmoid()