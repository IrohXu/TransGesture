# Toward Human Deictic Gesture Target Estimation [NeurIPS 2025]

#### [Xu Cao](https://www.irohxucao.com/), University of Illinois Urbana-Champaign

## Installation
* Create a conda virtual env and activate it.

  ```
  conda env create -f environment.yml
  conda activate GestureTarget
  ```
  or
  ```
  pip3 install -r requirements.txt
  ```
* Install [detectron2](https://github.com/facebookresearch/detectron2) , follow its [documentation](https://detectron2.readthedocs.io/en/latest/), or

  ```
  pip install "git+https://github.com/facebookresearch/detectron2.git@017abbfa5f2c2a2afa045200c2af9ccf2fc6227f#egg=detectron2"
  ```

## Train/Eval
### Pre-training/Fine-tuning/Testing Dataset Preprocessing

You should prepare GazeFollow and GestureTarget for training.

* Get [GazeFollow](https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?dl=0).

* Get GestureTarget-v1. Coming Soon.

Check `./configs/common/dataloader` to modify DATA_ROOT for Gaze Modeling.     
Check `./configs/common_gesture/dataloader` to modify DATA_ROOT for Gesture Modeling.     

### Pretrained Model

* Get [DINOv2](https://github.com/facebookresearch/dinov2) pretrained ViT-S/ViT-B/ViT-L/ViT-G.
* Or you could download and preprocess pretrained weights by

  ```
  mkdir pretrained && cd pretrained
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
  ```
* Preprocess the model weights with `./scripts/convert_pth.py` to fit Detectron2 format.      

### Finetune with GestureTarget
  ```
  python -u tools/train.py --config-file ./configs/gesture_jointcrossfusion_vit_small.py --num-gpu 2
  ```

* TODO: Implementation of DINOv3

## Evaluation

```
python tools/eval_on_gesture_target.py --config_file ./configs/gesture_basefusion_vit_small.py --model_weights xxx
```

## Reference

```
@inproceedings{caotoward,
  title={Toward Human Deictic Gesture Target Estimation},
  author={Cao, Xu and Virupaksha, Pranav and Lee, Sangmin and Lai, Bolin and Jia, Wenqi and Chen, Jintai and Rehg, James Matthew},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
  year={2025}
}
```

## Acknowledgement

Our implementation is based on [ViTGaze](https://github.com/hustvl/ViTGaze), [Gaze-LLE](https://github.com/fkryan/gazelle), and GazeAnywhere (coming soon). Thanks for their remarkable contribution and released code! If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.

