from .common_gesture.dataloader import dataloader
from .common_gesture.gesture_jointcrossfusion import model
from .common_gesture.optimizer import optimizer
from .common_gesture.scheduler import lr_multiplier
from .common_gesture.train import train
from os.path import join, basename
from torch.cuda import device_count


num_gpu = device_count()
ins_per_iter = 32
len_dataset = 15317
num_epoch = 50
# dataloader
dataloader = dataloader.gesture_target
dataloader.train.batch_size = ins_per_iter // num_gpu
dataloader.train.num_workers = dataloader.val.num_workers = 14
dataloader.train.distributed = num_gpu > 1
dataloader.train.rand_rotate = 0.5
dataloader.train.rand_lsj = 0.5
model.image_size = dataloader.train.input_size = dataloader.val.input_size = 518
model.output_size = dataloader.train.output_size = dataloader.val.output_size = 128

dataloader.val.batch_size = 32
dataloader.val.distributed = False
# train
train.init_checkpoint = "./output/gazefollow_vit_small/model_final.pth"
train.output_dir = join("./output", basename(__file__).split(".")[0])
train.max_iter = len_dataset * num_epoch // ins_per_iter
train.log_period = len_dataset // (ins_per_iter * 100)
train.checkpointer.max_to_keep = 3
train.checkpointer.period = len_dataset // ins_per_iter
train.seed = 0
# optimizer
optimizer.lr = 5e-4
optimizer.betas = (0.9, 0.99)
lr_multiplier.scheduler.typ = "cosine"
lr_multiplier.scheduler.start_value = 1
lr_multiplier.scheduler.end_value = 0.1
lr_multiplier.warmup_length = 1e-2
# model
model.backbone.name = "dinov2_small"
model.backbone.return_softmax_attn = True
model.backbone.out_attn = [2, 5, 8, 11]
model.backbone.use_cls_token = True
model.backbone.use_mask_token = True
model.criterion.use_focal_loss = True
model.device = "cuda"
model.freeze_backbone = True
model.freeze_gaze_branch = True
model.patch_size = 14
model.gaze_num_layers = 3
model.gesture_num_layers = 3
model.gaze_dim = model.fusion.gaze_dim = 256
model.gesture_dim = model.fusion.gesture_dim = 256
model.fusion.dim = 256
