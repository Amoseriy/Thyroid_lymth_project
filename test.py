#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/8 19:02
import SimpleITK as sitk
import numpy as np
import os

def get_label_numbers(lab_path):
    label = sitk.ReadImage(lab_path)
    label_array = sitk.GetArrayFromImage(label)
    unique_labels = np.unique(label_array)
    return unique_labels

# 设置要读取的目录路径
directory_path = r"G:\Program\DATABASE\2023\Xiang\ChenShuShu"

# 遍历目录及其子目录中的所有文件
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.seg.nrrd'):
            file_path = os.path.join(root, file)
            # print(f"Processing file: {file_path}")
            unique_labels = get_label_numbers(file_path)
            print(f"File: {file_path}, Unique Labels: {unique_labels}")