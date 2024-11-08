#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser
# @date: 2024/11/5 21:52

#问题出现在代码处理中 同一切片和同一段(segment_idx) 存在多个连通区域时，会覆盖之前的矩形掩膜，导致部分数据丢失。

import os
import nrrd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.measure import label, regionprops
import shutil


def process_nrrd_file(nrrd_path, label_class):
    data, header = nrrd.read(nrrd_path)
    label_info_by_plane = {}

    # 确定数据的维度
    if data.ndim == 4:
        # 处理4D数据，假设形状为 (segments, x, y, z)
        num_segments = data.shape[0]  # 即为标签数（LN数）
        x_size, y_size, z_slices = data.shape[1], data.shape[2], data.shape[3]
        for z_value in range(z_slices):
            for segment_idx in range(num_segments):
                z_slice_data = data[segment_idx, :, :, z_value]
                # 连通区域分析
                labeled_image = label(z_slice_data)
                regions = regionprops(labeled_image)
                # 初始化一个累积掩膜
                accumulated_mask = np.zeros_like(z_slice_data, dtype=np.uint8)  # 修改1：初始化累积掩膜
                label_info = []
                for region in regions:
                    min_row, min_col, max_row, max_col = region.bbox
                    # 在累积掩膜上标记矩形区域
                    accumulated_mask[min_row:max_row, min_col:max_col] = 1  # 修改2：在累积掩膜上标记矩形区域
                    # 保存边界框信息（可选）
                    label_info.append([label_class, min_row, min_col, max_row, max_col])
                # 将累积掩膜赋值回原数据
                data[segment_idx, :, :, z_value] = accumulated_mask  # 修改3：将累积掩膜赋值回原数据
                if label_info:
                    if z_value not in label_info_by_plane:
                        label_info_by_plane[z_value] = []
                    label_info_by_plane[z_value].extend(label_info)
    elif data.ndim == 3:
        x_size, y_size, z_slices = data.shape
        for z_value in range(z_slices):
            z_slice_data = data[:, :, z_value]
            # 连通区域分析
            labeled_image = label(z_slice_data)
            regions = regionprops(labeled_image)
            # 初始化一个累积掩膜
            accumulated_mask = np.zeros_like(z_slice_data, dtype=np.uint8)  # 修改1：初始化累积掩膜
            label_info = []
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                # 在累积掩膜上标记矩形区域
                accumulated_mask[min_row:max_row, min_col:max_col] = 1  # 修改2：在累积掩膜上标记矩形区域
                # 保存边界框信息（可选）
                label_info.append([label_class, min_row, min_col, max_row, max_col])
            # 赋值回原数据
            data[:, :, z_value] = accumulated_mask  # 修改3：将累积掩膜赋值回原数据
            if label_info:
                if z_value not in label_info_by_plane:
                    label_info_by_plane[z_value] = []
                label_info_by_plane[z_value].extend(label_info)
    else:
        raise ValueError("Unexpected data dimensions")

    return data, header


def process_file(nrrd_path: str, save_dir):
    file_name = os.path.basename(nrrd_path)
    label_class = 0 if '_B' in file_name else 1
    folder_name = os.path.basename(os.path.dirname(nrrd_path))
    new_data, header = process_nrrd_file(nrrd_path, label_class)
    new_file_path = os.path.join(save_dir, file_name)
    nrrd.write(new_file_path, new_data, header)
    print(f"{new_file_path} has been written.")


def process_folder(folder_path: str, save_dir):
    nrrd_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.seg.nrrd'):
                nrrd_files.append(os.path.join(root, file))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, nrrd_path, save_dir) for nrrd_path in nrrd_files]
        for future in as_completed(futures):
            future.result()


def classify_txt_files(folder_path):
    for txt_file in os.listdir(folder_path):
        if txt_file == "classes.txt":
            continue
        patient_name = txt_file.split('_')[0]
        patient_folder = os.path.join(folder_path, patient_name)
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)
        old_file_path = os.path.join(folder_path, txt_file)
        new_file_path = os.path.join(patient_folder, txt_file)
        os.rename(old_file_path, new_file_path)


# 设置文件夹路径(根目录即可，递归处理子文件夹)
input_folder = r"G:\Program\DATABASE"
output_label_folder = r"G:\Program\DATABASE_RECT"

# 创建保存文件夹
os.makedirs(output_label_folder, exist_ok=True)

# 处理文件夹
process_folder(input_folder, output_label_folder)
classify_txt_files(output_label_folder)