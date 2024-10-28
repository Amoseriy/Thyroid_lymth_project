#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/12 下午11:20
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.io import read_image


class MyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        images = read_image(self.image_paths[idx])

        # 对图像进行预处理，转换成tensor
        if self.transform:
            image = self.transform(images)

        # 获取图像label，并转换成tensor
        labels = torch.tensor(self.labels[idx], dtype=torch.int64)
        return image, labels


def get_my_dataset(ori_path):
    """
    :param ori_path: benign和malignant图片文件夹所在文件夹
    :return:
    """
    image_paths = []
    label_list = []
    be_path = os.path.join(ori_path, "benign/")
    mal_path = os.path.join(ori_path, "malignant/")

    # 写入benign图片路径和label
    be_images = os.listdir(be_path)
    for be_image in be_images:
        be_image_path = os.path.join(be_path, be_image).replace("\\", "/")
        label = int(0)

        image_paths.append(be_image_path)
        label_list.append(label)

    # 写入malignant图片路径和label
    mal_images = os.listdir(mal_path)
    for mal_image in mal_images:
        mal_image_path = os.path.join(mal_path, mal_image).replace("\\", "/")
        label = int(1)
        image_paths.append(mal_image_path)
        label_list.append(label)

    # 定义图像的预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToPILImage(),  # 将read_image()读入的图片tensor转换成PIL图像，以便Grayscale()使用
        transforms.Grayscale(num_output_channels=1),  # 转换成单通道图像
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = MyDataset(image_paths, label_list, transform)
    return dataset


# 把图片对应的tensor调整维度，并显示
def tensor2img(img_tensor):
    img_np = img_tensor.numpy()
    img = np.transpose(img_np, [1, 2, 0])
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    my_dataset = get_my_dataset("../output/")
    # print(len(my_dataset))

    # 分割成训练集和预测集
    n_train = int(len(my_dataset) * 0.8)
    n_val = len(my_dataset) - n_train
    ds_train, ds_val = random_split(my_dataset, [n_train, n_val])
    print(len(ds_train), len(ds_val))

    train_dataloader = DataLoader(ds_train, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=8, shuffle=True)

    for img, label in train_dataloader:
        print(img.shape)
        print(label[0])
        tensor2img(img[0, :, :, :])
        break
