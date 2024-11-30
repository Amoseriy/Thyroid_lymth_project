#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/24 15:44
import os
import json
import cv2
import numpy as np
from PIL import Image
import pathlib


def create_coco_json(image_folder, mask_folder, output_json_path):
    # 初始化COCO格式的字典
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 定义类别信息
    categories = [{"id": 1, "name": "benign", "supercategory": ""}, {"id": 2, "name": "malignant", "supercategory": ""}]
    coco_data["categories"] = categories

    mask_path_list = []
    for root, dirs, files in os.walk(mask_folder):
        for file in files:
            if file.endswith('.png'):
                mask_path = pathlib.Path(root, file)
                mask_path_list.append(mask_path)

    image_id = 1
    annotation_id = 1
    img_path_list = [f"{str(i.parents[1]).replace('_PNG', '_JPG')}\\{i.name.replace('.png', '.jpg')}"
                     for i in mask_path_list]
    img_path_list = list(set(img_path_list))

    path_dict = {}
    for img_path in img_path_list:
        img_path = img_path.replace('\\', '/')
        b_mask_name = (f"{'/'.join(img_path.split('/')[:-1]).replace('_JPG', '_PNG')}/benign/"
                       f"{img_path.split('/')[-1].replace('.jpg', '.png')}")
        m_mask_name = (f"{'/'.join(img_path.split('/')[:-1]).replace('_JPG', '_PNG')}/malignant/"
                       f"{img_path.split('/')[-1].replace('.jpg', '.png')}")
        # print(b_mask_name, m_mask_name)
        mask_path_list = []
        if os.path.exists(b_mask_name):
            mask_path_list.append(b_mask_name)
        if os.path.exists(m_mask_name):
            mask_path_list.append(m_mask_name)
        path_dict[img_path] = mask_path_list
    # print(path_dict)
    # with open('path_dict.json', 'w') as f:
    #     json.dump(path_dict, f, indent=4)
    # 遍历字典中的图像和掩膜文件
    for image_path, mask_paths in path_dict.items():
        # 获取图像信息
        image = Image.open(image_path)
        width, height = image.size

        # 创建图像条目
        image_entry = {
            "id": image_id,
            "file_name": image_path,
            "width": width,
            "height": height
        }
        coco_data["images"].append(image_entry)

        for mask_path in mask_paths:
            # 读取掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 获取掩码中的所有唯一值
            unique_values = np.unique(mask)

            for value in unique_values:
                if value == 0:  # 忽略背景
                    continue

                # 提取特定值的掩码
                category_mask = (mask == value).astype(np.uint8) * 255
                contours, _ = cv2.findContours(category_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    # 计算边界框
                    x, y, w, h = cv2.boundingRect(contour)

                    # 计算多边形坐标
                    segmentation = contour.flatten().tolist()

                    # 根据路径关键字设置类别ID
                    if 'benign' in mask_path:
                        category_id = 1
                    elif 'malignant' in mask_path:
                        category_id = 2
                    else:
                        raise ValueError("Unknown category in mask path: {}".format(mask_path))

                    # 创建注释条目
                    annotation_entry = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": [segmentation],
                        "area": cv2.contourArea(contour),
                        "bbox": [x, y, w, h],
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation_entry)
                    annotation_id += 1

        image_id += 1

    # 写入JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


if __name__ == '__main__':
    image_folder = 'G:/Program/DATABASE_JPG'
    mask_folder = 'G:/Program/DATABASE_PNG'
    # 使用函数
    create_coco_json(image_folder, mask_folder, 'output.json')