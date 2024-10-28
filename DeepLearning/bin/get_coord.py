#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/10 下午10:59
import os
import cv2


def get_coord(file_path):
    """
    :param file_path: txt文件的路径
    :return: 返回（label，x1，y1，x2，y2）列表
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析每行的标签和坐标
    annotations = []
    for line in lines:
        parts = line.strip().split()
        label = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        annotations.append((label, x_center, y_center, width, height))

    # 假设原始图像为 image.jpg
    image_path = file_path.replace("TXT", "JPG").replace(".txt", ".jpg")
    # print(image_path)
    image = cv2.imread(image_path)
    img_height, img_width = 224.0, 224.0  # image.shape[:2]

    coord = []  # 定义返回的坐标列表
    for label, x_center, y_center, width, height in annotations:
        # 转换为图像上的像素坐标
        x_center_pixel = int(x_center * img_width)
        y_center_pixel = int(y_center * img_height)
        width_pixel = int(width * img_width)
        height_pixel = int(height * img_height)

        # 计算左上角和右下角的坐标
        x1 = max(0, x_center_pixel - width_pixel // 2 - 2)
        y1 = max(0, y_center_pixel - height_pixel // 2 - 2)
        x2 = min(img_width, x_center_pixel + width_pixel // 2 + 2)
        y2 = min(img_height, y_center_pixel + height_pixel // 2 + 2)

        coord.append((label, x1, y1, x2, y2))
    return coord


if __name__ == '__main__':
    path = "../TXT/Xiang/BaiMaoAn/BaiMaoAn121.txt"
    co_ord = get_coord(path)
    print(co_ord)
