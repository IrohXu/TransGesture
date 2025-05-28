import math
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF


class Augmentation:
    def __init__(self, p: float) -> None:
        self.p = p

    def transform(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        raise NotImplementedError

    def __call__(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        if np.random.random_sample() < self.p:
            return self.transform(image, target_mask, bbox, head_bboxes, target_bboxes, size)
        return image, target_mask, bbox, head_bboxes, target_bboxes, size


class AugmentationList:
    def __init__(self, augmentations: List[Augmentation]) -> None:
        self.augmentations = augmentations

    def __call__(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        for aug in self.augmentations:
            image, target_mask, bbox, head_bboxes, target_bboxes, size = aug(image, target_mask, bbox, head_bboxes, target_bboxes, size)
        return image, target_mask, bbox, head_bboxes, target_bboxes, size


class BoxJitter(Augmentation):
    # Jitter (expansion-only) bounding box size
    def __init__(self, p: float, expansion: float = 0.2) -> None:
        super().__init__(p)
        self.expansion = expansion

    def transform(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        x_min, y_min, x_max, y_max = bbox
        width, height = size
        k = np.random.random_sample() * self.expansion
        x_min = np.clip(x_min - k * abs(x_max - x_min), 0, width - 1)
        y_min = np.clip(y_min - k * abs(y_max - y_min), 0, height - 1)
        x_max = np.clip(x_max + k * abs(x_max - x_min), 0, width - 1)
        y_max = np.clip(y_max + k * abs(y_max - y_min), 0, height - 1)
        
        if head_bboxes != None:
            for i, bbox in enumerate(head_bboxes):
                hx_min = np.clip(bbox[0] - k * abs(bbox[2] - bbox[0]), 0, width - 1)
                hy_min = np.clip(bbox[1] - k * abs(bbox[3] - bbox[1]), 0, height - 1)
                hx_max = np.clip(bbox[2] + k * abs(bbox[2] - bbox[0]), 0, width - 1)
                hy_max = np.clip(bbox[3] + k * abs(bbox[3] - bbox[1]), 0, height - 1)
                head_bboxes[i] = (hx_min, hy_min, hx_max, hy_max)
        
        target_bboxes_new = []
        for bbox in target_bboxes:
            tx_min = np.clip(bbox[0] - k * abs(bbox[2] - bbox[0]), 0, width - 1)
            ty_min = np.clip(bbox[1] - k * abs(bbox[3] - bbox[1]), 0, height - 1)
            tx_max = np.clip(bbox[2] + k * abs(bbox[2] - bbox[0]), 0, width - 1)
            ty_max = np.clip(bbox[3] + k * abs(bbox[3] - bbox[1]), 0, height - 1)
            target_bboxes_new.append((tx_min, ty_min, tx_max, ty_max))
            
        return image, target_mask, (x_min, y_min, x_max, y_max), head_bboxes, target_bboxes_new, size


class RandomCrop(Augmentation):
    def __init__(self, p: float) -> None:
        super().__init__(p)

    def transform(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        x_min, y_min, x_max, y_max = bbox
        width, height = size
        
        tx_min_list = [] 
        tx_max_list = []
        ty_min_list = []
        ty_max_list = []
        for tbbox in target_bboxes:
            tx_min_list.append(tbbox[0])
            ty_min_list.append(tbbox[1])
            tx_max_list.append(tbbox[2])
            ty_max_list.append(tbbox[3])
        # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
        crop_x_min = np.min(tx_min_list + tx_max_list + [x_min, x_max])
        crop_y_min = np.min(ty_min_list + ty_max_list + [y_min, y_max])
        crop_x_max = np.max(tx_min_list + tx_max_list + [x_min, x_max])
        crop_y_max = np.max(ty_min_list + ty_max_list + [y_min, y_max])

        # Randomly select a random top left corner
        crop_x_min = np.random.uniform(0, crop_x_min)
        crop_y_min = np.random.uniform(0, crop_y_min)

        # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
        crop_width_min = crop_x_max - crop_x_min
        crop_height_min = crop_y_max - crop_y_min
        crop_width_max = width - crop_x_min
        crop_height_max = height - crop_y_min

        # Randomly select a width and a height
        crop_width = np.random.uniform(crop_width_min, crop_width_max)
        crop_height = np.random.uniform(crop_height_min, crop_height_max)

        # Round to integers
        crop_y_min, crop_x_min, crop_height, crop_width = map(
            int, map(round, (crop_y_min, crop_x_min, crop_height, crop_width))
        )

        # Crop it
        image = TF.crop(image, crop_y_min, crop_x_min, crop_height, crop_width)
        target_mask = TF.crop(target_mask, crop_y_min, crop_x_min, crop_height, crop_width)
        
        # convert coordinates into the cropped frame
        x_min, y_min, x_max, y_max = (
            x_min - crop_x_min,
            y_min - crop_y_min,
            x_max - crop_x_min,
            y_max - crop_y_min,
        )
        
        if head_bboxes != None:
            for i, bbox in enumerate(head_bboxes):
                hx_min = bbox[0] - crop_x_min
                hy_min = bbox[1] - crop_y_min
                hx_max = bbox[2] - crop_x_min
                hy_max = bbox[3] - crop_y_min
                head_bboxes[i] = (hx_min, hy_min, hx_max, hy_max)

        target_bboxes_new = []
        for tbbox in target_bboxes:
            tx_min = tbbox[0] - crop_x_min
            ty_min = tbbox[1] - crop_y_min
            tx_max = tbbox[2] - crop_x_min
            ty_max = tbbox[3] - crop_y_min
            target_bboxes_new.append((tx_min, ty_min, tx_max, ty_max))
        
        return (
            image,
            target_mask,
            (x_min, y_min, x_max, y_max),
            head_bboxes,
            target_bboxes_new,
            (crop_width, crop_height),
        )


class RandomFlip(Augmentation):
    def __init__(self, p: float) -> None:
        super().__init__(p)

    def transform(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        target_mask = target_mask.transpose(Image.FLIP_LEFT_RIGHT)
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max = size[0] - x_max, size[0] - x_min
        
        if head_bboxes != None:
            for i, bbox in enumerate(head_bboxes):
                hx_min = size[0] - bbox[2]
                hy_min = bbox[1]
                hx_max = size[0] - bbox[0]
                hy_max = bbox[3]
                head_bboxes[i] = (hx_min, hy_min, hx_max, hy_max)
        
        target_bboxes_new = []
        for bbox in target_bboxes:
            tx_min = size[0] - bbox[2]
            ty_min = bbox[1]
            tx_max = size[0] - bbox[0]
            ty_max = bbox[3]
            target_bboxes_new.append((tx_min, ty_min, tx_max, ty_max))
        
        return image, target_mask, (x_min, y_min, x_max, y_max), head_bboxes, target_bboxes_new, size


class RandomRotate(Augmentation):
    def __init__(
        self, p: float, max_angle: int = 20, resample: int = Image.BILINEAR
    ) -> None:
        super().__init__(p)
        self.max_angle = max_angle
        self.resample = resample

    def _random_rotation_matrix(self):
        angle = (2 * np.random.random_sample() - 1) * self.max_angle
        angle = -math.radians(angle)
        return [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

    @staticmethod
    def _transform(x, y, matrix):
        return (
            matrix[0] * x + matrix[1] * y + matrix[2],
            matrix[3] * x + matrix[4] * y + matrix[5],
        )

    @staticmethod
    def _inv_transform(x, y, matrix):
        x, y = x - matrix[2], y - matrix[5]
        return matrix[0] * x + matrix[3] * y, matrix[1] * x + matrix[4] * y

    def transform(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        x_min, y_min, x_max, y_max = bbox
        width, height = size
        rot_mat = self._random_rotation_matrix()

        # Calculate offsets
        rot_center = (width / 2.0, height / 2.0)
        rot_mat[2], rot_mat[5] = self._transform(
            -rot_center[0], -rot_center[1], rot_mat
        )
        rot_mat[2] += rot_center[0]
        rot_mat[5] += rot_center[1]
        xx = []
        yy = []
        for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
            x, y = self._transform(x, y, rot_mat)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))
        rot_mat[2], rot_mat[5] = self._transform(
            -(nw - width) / 2.0, -(nh - height) / 2.0, rot_mat
        )

        image = image.transform((nw, nh), Image.AFFINE, rot_mat, self.resample)
        target_mask = target_mask.transform((nw, nh), Image.AFFINE, rot_mat, self.resample)

        xx = []
        yy = []
        for x, y in (
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max),
        ):
            x, y = self._inv_transform(x, y, rot_mat)
            xx.append(x)
            yy.append(y)
        x_max, x_min = min(max(xx), nw), max(min(xx), 0)
        y_max, y_min = min(max(yy), nh), max(min(yy), 0)
        
        if head_bboxes != None:
            for i, bbox in enumerate(head_bboxes):
                hx_min, hy_min, hx_max, hy_max = bbox
                xx = []
                yy = []
                for x, y in (
                    (hx_min, hy_min),
                    (hx_min, hy_max),
                    (hx_max, hy_min),
                    (hx_max, hy_max),
                ):
                    x, y = self._inv_transform(x, y, rot_mat)
                    xx.append(x)
                    yy.append(y)
                hx_max, hx_min = min(max(xx), nw), max(min(xx), 0)
                hy_max, hy_min = min(max(yy), nh), max(min(yy), 0)
                head_bboxes[i] = (hx_min, hy_min, hx_max, hy_max)
        
        target_bboxes_new = []
        for bbox in target_bboxes:
            tx_min, ty_min, tx_max, ty_max = bbox
            xx = []
            yy = []
            for x, y in (
                (tx_min, ty_min),
                (tx_min, ty_max),
                (tx_max, ty_min),
                (tx_max, ty_max),
            ):
                x, y = self._inv_transform(x, y, rot_mat)
                xx.append(x)
                yy.append(y)
            tx_max, tx_min = min(max(xx), nw), max(min(xx), 0)
            ty_max, ty_min = min(max(yy), nh), max(min(yy), 0)
            target_bboxes_new.append((tx_min, ty_min, tx_max, ty_max))

        return (
            image,
            target_mask,
            (x_min, y_min, x_max, y_max),
            head_bboxes,
            target_bboxes_new,
            (nw, nh),
        )


class ColorJitter(Augmentation):
    def __init__(
        self,
        p: float,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
    ) -> None:
        super().__init__(p)
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def transform(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        return self.color_jitter(image), target_mask, bbox, head_bboxes, target_bboxes, size


class RandomLSJ(Augmentation):
    def __init__(self, p: float, min_scale: float = 0.5) -> None:
        super().__init__(p)
        self.min_scale = min_scale

    def transform(
        self,
        image: Image,
        target_mask: Image,
        bbox: Tuple[float],
        head_bboxes: List[Tuple[float]],
        target_bboxes: List[Tuple[float]],
        size: Tuple[int],
    ):
        x_min, y_min, x_max, y_max = bbox
        width, height = size

        scale = self.min_scale + np.random.random_sample() * (1 - self.min_scale)
        nh, nw = int(height * scale), int(width * scale)

        image = TF.resize(image, (nh, nw))
        image = ImageOps.expand(image, (0, 0, width - nw, height - nh))
        
        target_mask = TF.resize(target_mask, (nh, nw))
        target_mask = ImageOps.expand(target_mask, (0, 0, width - nw, height - nh))

        x_min, y_min, x_max, y_max = (
            x_min * scale,
            y_min * scale,
            x_max * scale,
            y_max * scale,
        )
        
        if head_bboxes != None:
            for i, bbox in enumerate(head_bboxes):
                hx_min = bbox[0] * scale
                hy_min = bbox[1] * scale
                hx_max = bbox[2] * scale
                hy_max = bbox[3] * scale
                head_bboxes[i] = (hx_min, hy_min, hx_max, hy_max)
        
        target_bboxes_new = []
        for bbox in target_bboxes:
            tx_min = bbox[0] * scale
            ty_min = bbox[1] * scale
            tx_max = bbox[2] * scale
            ty_max = bbox[3] * scale
            target_bboxes_new.append((tx_min, ty_min, tx_max, ty_max))
        
        return image, target_mask, (x_min, y_min, x_max, y_max), head_bboxes, target_bboxes_new, size
