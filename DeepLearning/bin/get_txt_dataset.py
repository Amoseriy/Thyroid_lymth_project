#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/11 下午12:38
import os
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from .get_coord import get_coord
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# 定义自定义数据集类
class FastRCNNDataset(Dataset):
    def __init__(self, image_paths, rois_labels, transform=None):
        self.image_paths = image_paths
        self.rois_labels = rois_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('L')  # 单通道图像

        # 对图像进行预处理
        if self.transform:
            image = self.transform(image)

        # 获取该图像对应的RoIs和标签
        image_rois_labels = self.rois_labels[idx]
        rois = []
        labels = []
        for item in image_rois_labels:
            labels.append(item[0])
            rois.append(item[1:])
        rois = torch.tensor(rois, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, rois, labels


def get_txt_path(path):
    total_txt_path = []
    researchers = os.listdir(path)
    for researcher in researchers:
        researcher_path = os.path.join(path, researcher)
        patients = os.listdir(researcher_path)
        for patient in patients:
            if patient == "classes.txt":
                continue
            patient_path = os.path.join(researcher_path, patient)
            txt_paths = os.listdir(patient_path)
            for txt in txt_paths:
                if txt == "classes.txt":
                    continue
                txt_path = os.path.join(patient_path, txt).replace("\\", "/")
                total_txt_path.append(txt_path)
    return total_txt_path


def get_dataset(ori_txt_path):
    # ori_txt_path = f"../{ori_txt_path}"
    txt_paths = get_txt_path(ori_txt_path)  # 获取txt路径列表
    # 获取图像列表
    image_paths = [char.replace("TXT", "JPG").replace(".txt", ".jpg") for char in txt_paths]
    # print(image_paths)

    # 示例RoIs和标签，格式为 [(label, x1, y1, x2, y2)]
    rois_labels = []
    for txt_path in txt_paths:
        coord = get_coord(txt_path)
        rois_labels.append(coord)
    # print(rois_labels)

    # 定义图像的预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 创建数据集实例
    dataset = FastRCNNDataset(image_paths=image_paths, rois_labels=rois_labels, transform=transform)

    return dataset


if __name__ == '__main__':
    dataset = get_dataset("../TXT/")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for images, rois, labels in dataloader:
        images = list(image.to(device) for image in images)
        targets = []
        for roi, label in zip(rois, labels):
            targets.append({
                'boxes': roi.to(device),
                'labels': label.to(device)
            })
        print(targets[0])
        print(images[0])
        break
