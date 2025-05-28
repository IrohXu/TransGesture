from detectron2.config import LazyCall as L

from modeling import backbone, patch_attention, meta_arch, head, criterion, fusion

model = L(meta_arch.GestureFusionMapper)()
model.backbone = L(backbone.build_backbone)(
    name="small"
)
model.fusion = L(fusion.build_fusion_module)(
    name="JointCrossAttentionFusion",
    gesture_dim=256,
    gaze_dim=256,
    dim=256,
    num_layers=3
)

model.criterion = L(criterion.GestureFMCriterion)()
model.device = "cuda"
