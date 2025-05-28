from detectron2.config import LazyCall as L

from modeling import backbone, patch_attention, meta_arch, head, criterion, fusion

model = L(meta_arch.GestureBaselineMapper)()
model.backbone = L(backbone.build_backbone)(
    name="small"
)
model.criterion = L(criterion.GestureFMCriterion)()
model.device = "cuda"
