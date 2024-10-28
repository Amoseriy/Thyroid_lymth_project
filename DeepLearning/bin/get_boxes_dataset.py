#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/17 下午5:51
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes


class GetBoxesDataset(Dataset):
    def __init__(self, root: str, transforms, malign=True):
        """
        :param root: 文件所在根目录
        :param transforms: 图片变换方式
        :param malign: 默认导入恶性，良性为False
        """
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.malign = malign
        # load all image files, sorting them to
        # ensure that they are aligned
        if self.malign:
            self.png_path: str = "Total/malign/"
        else:
            self.png_path: str = "Total/benign/"
        self.masks = list(sorted(os.listdir(os.path.join(root, self.png_path))))
        self.imgs = [
            img.replace("_L_B", "").replace("_L_M", "").replace("_R_B", "")
            .replace("_R_M", "").replace("png", "jpg") for img in self.masks
        ]

        # Filter out images without valid targets
        valid_samples = []
        for img, mask in zip(self.imgs, self.masks):
            mask_path = os.path.join(self.root, self.png_path, mask)
            mask_img = read_image(mask_path)
            mask_img = F.convert_image_dtype(mask_img, dtype=torch.float)
            obj_ids = torch.unique(mask_img)
            if len(obj_ids) > 1:  # If there are more than just the background
                valid_samples.append((img, mask))

        self.imgs, self.masks = zip(*valid_samples)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "JPG", self.imgs[idx])
        mask_path = os.path.join(self.root, self.png_path, self.masks[idx])

        img = read_image(img_path)
        mask = read_image(mask_path)

        img = F.convert_image_dtype(img, dtype=torch.float)
        mask = F.convert_image_dtype(mask, dtype=torch.float)

        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]
        # obj_ids[:, None, None] 使用了 None 来扩展 obj_ids 的维度，将其从形状 (N,) 扩展为 (N, 1, 1)
        # 这使得 obj_ids 可以在后续操作中与 mask 进行广播
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)

        # 定义边界框扩展的像素值
        expand_pixels = 3
        # 扩展边界框
        expanded_boxes = boxes.clone()
        expanded_boxes[:, 0] -= expand_pixels  # x_min
        expanded_boxes[:, 1] -= expand_pixels  # y_min
        expanded_boxes[:, 2] += expand_pixels  # x_max
        expanded_boxes[:, 3] += expand_pixels  # y_max

        # there is only one class
        if self.malign:
            labels = torch.ones((masks.shape[0],), dtype=torch.int64)
        else:
            labels = torch.zeros((masks.shape[0],), dtype=torch.int64)

        targets = {"boxes": expanded_boxes, "labels": labels}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, targets

    def __len__(self):
        return len(self.imgs)
