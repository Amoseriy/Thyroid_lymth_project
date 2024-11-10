#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser
# @date: 2024/11/5 21:52

import os
from collections import Counter
import nrrd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.measure import label as ski_label
from skimage.measure import regionprops as ski_regionprops
import shutil


def get_three_d_data(three_d_data):
    x_size, y_size, z_slices = three_d_data.shape
    for z_value in range(z_slices):
        z_slice_data = three_d_data[:, :, z_value]
        labeled_image, num_labels = ski_label(z_slice_data, return_num=True, connectivity=1)
        regions = ski_regionprops(labeled_image)
        if num_labels != 0:
            # print(f"Number of images: {z_value}")
            # 初始化一个累积掩膜
            # accumulated_mask = np.zeros_like(z_slice_data, dtype=data.dtype)
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                region_label_num = region.label
                sub_matrix = z_slice_data[min_row:max_row, min_col:max_col]
                # 统计bbox内的非0像素值
                non_zero_values = sub_matrix[sub_matrix != 0]
                # 统计非0像素值的频率
                value_counts = Counter(non_zero_values)
                # print(value_counts)
                # 找到出现次数第一多的像素值及其出现的次数
                label_value, most_common_count = value_counts.most_common(1)[0]
                print(f"Label value: {label_value}")
                sub_matrix[:] = label_value
                z_slice_data[min_row:max_row, min_col:max_col] = sub_matrix
                # print(z_slice_data)
                three_d_data[:, :, z_value] = z_slice_data
    return three_d_data



def process_nrrd_file(nrrd_path):
    data, header = nrrd.read(nrrd_path)
    print(nrrd_path)
    label_info_by_plane = {}

    # 确定数据的维度
    if data.ndim == 4:
        # 处理4D数据，假设形状为 (n, x, y, z)
        data_dict = {}  # 用于保存每个切片的处理结果
        for i in range(data.shape[0]):
            new_arr = f"data_{i}"
            data_dict[new_arr] = get_three_d_data(data[i, :, :, :])
        # 合并切片数据
        for n in range(data.shape[0]):
            data[n, :, :, :] = data_dict[f"data_{n}"]

    elif data.ndim == 3:
        data = get_three_d_data(data)
    else:
        raise ValueError("Unexpected data dimensions")

    return data, header


def process_file(nrrd_path: str, input_root: str, output_root: str):
    # 计算相对路径
    rel_path = os.path.relpath(nrrd_path, input_root)
    # 构建输出路径
    output_path = os.path.join(output_root, rel_path)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    new_data, header = process_nrrd_file(nrrd_path)
    nrrd.write(output_path, new_data, header)
    print(f"{output_path} has been written.")


def process_folder(input_folder: str, output_folder: str):
    nrrd_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.seg.nrrd'):
                nrrd_files.append(os.path.join(root, file))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, nrrd_path, input_folder, output_folder)
            for nrrd_path in nrrd_files
        ]
        for future in as_completed(futures):
            future.result()


def classify_txt_files(folder_path:str):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "classes.txt" or not file.endswith('.txt'):
                continue
            patient_name = file.split('_')[0]
            patient_folder = os.path.join(root, patient_name)
            if not os.path.exists(patient_folder):
                os.makedirs(patient_folder)
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(patient_folder, file)
            os.rename(old_file_path, new_file_path)


# 设置文件夹路径(根目录即可，递归处理子文件夹)
input_folder = r"G:\Program\DATABASE"
output_label_folder = r"G:\Program\DATABASE_RECT"

# 创建保存文件夹
os.makedirs(output_label_folder, exist_ok=True)

# 处理文件夹
process_folder(input_folder, output_label_folder)
# classify_txt_files(output_label_folder)


