from .gesture_jointcrossfusion_vit_base import dataloader, model, optimizer, lr_multiplier, train

model.criterion.mask_loss_positive_only = True
