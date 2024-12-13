#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/3/16 10:46
import os
import json
import shutil
from collections import Counter

from skimage.measure import label as ski_label
from skimage.measure import regionprops as ski_regionprops
import numpy as np
import cv2


with open("G:\Program\DeepLearning\img_mask_path.json", "r") as f:
    path_dict = json.load(f)

def mask2yolo(fn_mask_paths):
    image_height, image_width = cv2.imread(img_path).shape[:2]
    # print(len(masks))
    inner_all_bboxes = []
    for mask_path in fn_mask_paths:
        class_id = 0 if "benign" in mask_path else 1
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        labeled_image, num_labels = ski_label(mask, return_num=True, connectivity=1)
        regions = ski_regionprops(labeled_image)
        bbox_list = []
        for region in regions:
            # 获取边界框 (min_row, min_col, max_row, max_col)
            minr, minc, maxr, maxc = region.bbox
            # 将边界框坐标转换为YOLO格式 (归一化)
            center_x = (minc + (maxc - minc) / 2) / image_width
            center_y = (minr + (maxr - minr) / 2) / image_height
            bbox_width = (maxc - minc) / image_width
            bbox_height = (maxr - minr) / image_height
            # 添加到bboxes列表
            bbox_list.append([class_id, center_x, center_y, bbox_width, bbox_height])
            # print(bbox_list[-1])
        inner_all_bboxes.extend(bbox_list)
    return inner_all_bboxes

output_file = r"G:\Program\DATABASE_YOLO\label"
for img_path in path_dict.keys():
    mask_paths = path_dict[img_path]
    all_bboxes = mask2yolo(mask_paths)
    file_name = f"{os.path.basename(img_path)[:-4]}.txt"
    output_file_path = os.path.join(output_file, file_name)
    # 写入YOLO格式的txt文件
    with open(output_file_path, 'w') as f:
        print(f"Writing {output_file_path}...")
        for bbox in all_bboxes:
            line = ' '.join([str(item) for item in bbox]) + '\n'
            f.write(line)


