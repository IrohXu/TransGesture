# Toward Human Deictic Gesture Target Estimation

## Installation
* Create a conda virtual env and activate it.

  ```
  conda env create -f environment.yml
  conda activate GestureTarget
  ```
* Install [detectron2](https://github.com/facebookresearch/detectron2) , follow its [documentation](https://detectron2.readthedocs.io/en/latest/), or

  ```
  pip install "git+https://github.com/facebookresearch/detectron2.git@017abbfa5f2c2a2afa045200c2af9ccf2fc6227f#egg=detectron2"
  ```


## Train/Eval
### Pre-training/Fine-tuning/Testing Dataset Preprocessing

You should prepare GazeFollow and GestureTarget for training.

* Get [GazeFollow](https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?dl=0).
* Use `./scripts/gen_gazefollow_head_masks.py` to generate head masks.

* Get [GestureTarget](Internal).

Check `./configs/common/dataloader` to modify DATA_ROOT for Gaze Modeling.     
Check `./configs/common_gesture/dataloader` to modify DATA_ROOT for Gaze Modeling.     

### Pretrained Model

* Get [DINOv2](https://github.com/facebookresearch/dinov2) pretrained ViT-S/ViT-B/ViT-L/ViT-G.
* Or you could download and preprocess pretrained weights by

  ```
  mkdir pretrained && cd pretrained
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
  ```
* Preprocess the model weights with `./scripts/convert_pth.py` to fit Detectron2 format.      
* Pre-train the model by
  
  ```
  python -u tools/train.py --config-file ./configs/gazefollow_gaze_vit_small.py --num-gpu 1
  ```

### Finetune with GestureTarget
  ```
  python -u tools/train.py --config-file ./configs/gesture_jointcrossfusion_vit_small.py --num-gpu 1
  ```


## Evaluation

```
python tools/eval_on_gesture_target.py --config_file ./configs/gesture_basefusion_vit_small.py --model_weights ./output/gesture_basefusion_vit_small/model_final.pth
```

## Reference

TODO
