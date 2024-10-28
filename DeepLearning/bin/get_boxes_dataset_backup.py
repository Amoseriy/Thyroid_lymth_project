#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/17 下午5:51
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
from get_zip_path import get_zip_path


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
        self.mal_png_path: str = "Total/malign/"
        self.ben_png_path: str = "Total/benign/"
        self.mal_masks = list(sorted(os.listdir(os.path.join(root, self.mal_png_path))))
        self.ben_masks = list(sorted(os.listdir(os.path.join(root, self.ben_png_path))))
        self.mal_imgs = [
            img.replace("_L_B", "").replace("_L_M", "").replace("_R_B", "")
            .replace("_R_M", "").replace("png", "jpg")
            for img in self.mal_masks
        ]
        self.ben_imgs = [
            img.replace("_L_B", "").replace("_L_M", "").replace("_R_B", "")
            .replace("_R_M", "").replace("png", "jpg")
            for img in self.ben_masks
        ]
        self.imgs = self.mal_imgs + self.ben_imgs
        self.masks = self.mal_masks + self.ben_masks
        # print(self.imgs[8000], self.masks[8000])
        # Filter out images without valid targets
        valid_samples = []
        for img, mask in zip(self.imgs, self.masks):
            if "_L_B" in mask or "_R_B" in mask:
                mask_path = os.path.join(self.root, self.ben_png_path, mask)
            else:
                mask_path = os.path.join(self.root, self.mal_png_path, mask)
            mask_img = read_image(mask_path)
            mask_img = F.convert_image_dtype(mask_img, dtype=torch.float)
            obj_ids = torch.unique(mask_img)
            if len(obj_ids) > 1:  # If there are more than just the background
                valid_samples.append((img, mask))

        self.imgs, self.masks = zip(*valid_samples)
        self.zip_path = get_zip_path(self.imgs, self.masks)
        # print(len(self.imgs), len(self.masks))
        # print(self.imgs[:10], self.masks[:10])
        # print(sum(["_L_M" in s for s in self.masks]))
        # print(sum(["_L_B" in s for s in self.masks]))
        # print(sum(["_R_M" in s for s in self.masks]))
        # print(sum(["_R_M" in s for s in self.masks]))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "JPG", self.imgs[idx])
        mask_paths = self.zip_path[self.imgs[idx]]

        mask_paths_list = []
        for mask_path in mask_paths:
            if "_L_B" in mask_path or "_R_B" in mask_path:
                mask_path = os.path.join(self.root, self.ben_png_path, self.masks[idx])
                mask_paths_list.append(mask_path)
            else:
                mask_path = os.path.join(self.root, self.mal_png_path, self.masks[idx])
                mask_paths_list.append(mask_path)

        img = read_image(img_path)
        img = F.convert_image_dtype(img, dtype=torch.float)

        box_list = []
        label_list = []
        for i in mask_paths_list:
            if "_L_B" in i or "_R_B" in i:
                label_list.append(torch.tensor([0]))
            else:
                label_list.append(torch.tensor([1]))
            mask = read_image(i)
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
            box_list.append(expanded_boxes)
        expanded_boxes = torch.cat(box_list, 0)
        labels = torch.cat(label_list, 0)

        targets = {"boxes": expanded_boxes, "labels": labels}
        print(targets)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, targets

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = GetBoxesDataset(root="../Data/", transforms=None, malign=True)
    for img, targets in dataset:
        print(img.shape)
        print(targets)
