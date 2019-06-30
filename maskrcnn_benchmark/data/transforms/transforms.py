# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomCrop():
    def __init__(self, w, h, s, min_area, min_visibility):
        self.w = w
        self.h = h
        self.s = s
        self.min_area = min_area
        self.min_visibility = min_visibility

    def __call__(self, image, target=None):
        bboxes = target.convert('xyxy').bbox
        box = bboxes[np.random.randint(bboxes.shape[0])]
        left, top, right, bottom = box

        im_w, im_h = image.size

        if (right - left) > self.w:
            x_from = max(0, left-self.s)
            x_to = x_from + 1
        else:
            x_from = max(0, right - self.w)
            x_to = min(im_w - self.w, left) + 1
        x = np.random.randint(int(x_from), int(x_to))

        if (bottom - top) > self.h:
            y_from = max(0, top-self.s)
            y_to = y_from + 1
        else:
            y_from = max(0, bottom - self.h)
            y_to = min(im_h - self.h, top)  + 1
        y = np.random.randint(int(y_from), int(y_to))

        crop = [x, y, x+self.w, y+self.h]

        new_image = image.crop(crop)
        cropped_bboxes = target.crop(crop)
        labels = cropped_bboxes.get_field('labels')
        areas = target.area()
        cropped_areas = cropped_bboxes.area()

        idx = (cropped_areas >= self.min_area) & ((cropped_areas / areas) >= self.min_visibility)

        new_boxes = cropped_bboxes.bbox[idx]
        new_labels = cropped_bboxes.get_field('labels')[idx]
        target = BoxList(new_boxes, (self.w, self.h), mode=target.mode)
        target.add_field('labels', torch.LongTensor(new_labels))
        return new_image, target


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target
