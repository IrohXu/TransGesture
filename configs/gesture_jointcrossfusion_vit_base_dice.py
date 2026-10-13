from .gesture_jointcrossfusion_vit_base import dataloader, model, optimizer, lr_multiplier, train

model.criterion.dice_weight = 100
