#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/10 下午4:50

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
        num_segments = data.shape[0] # 即为标签数（LN数）
        x_size, y_size, z_slices = data.shape[1], data.shape[2], data.shape[3]
        for z_value in range(z_slices):
            label_info = []
            for segment_idx in range(num_segments):
                z_slice_data = data[segment_idx, :, :, z_value]
                # 连通区域分析，label 函数会将相连的区域标记为同一标签。
                labeled_image = label(z_slice_data)
                # 计算每个连通区域的长度、宽度、中心点坐标、比例
                regions = regionprops(labeled_image)
                # 遍历、保存每个连通区域的长度、宽度、中心点坐标、比例
                for region in regions:
                    min_row, min_col, max_row, max_col = region.bbox
                    length = max_row - min_row
                    width = max_col - min_col
                    center_x = (min_row + max_row) / 2
                    center_y = (min_col + max_col) / 2
                    center_x_ratio = center_x / x_size
                    center_y_ratio = center_y / y_size
                    length_ratio = length / x_size
                    width_ratio = width / y_size
                    label_info.append([label_class, f"{center_x_ratio:.6f}", f"{center_y_ratio:.6f}",
                                       f"{length_ratio:.6f}", f"{width_ratio:.6f}"])
            if label_info:
                if z_value not in label_info_by_plane:
                    label_info_by_plane[z_value] = []
                label_info_by_plane[z_value].extend(label_info)
    elif data.ndim == 3:
        x_size, y_size, z_slices = data.shape
        for z_value in range(z_slices):
            label_info = []
            z_slice_data = data[:, :, z_value]
            # 连通区域分析
            labeled_image = label(z_slice_data)
            regions = regionprops(labeled_image)
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                length = max_row - min_row
                width = max_col - min_col
                center_x = (min_row + max_row) / 2
                center_y = (min_col + max_col) / 2
                center_x_ratio = center_x / x_size
                center_y_ratio = center_y / y_size
                length_ratio = length / x_size
                width_ratio = width / y_size
                label_info.append([label_class, f"{center_x_ratio:.6f}", f"{center_y_ratio:.6f}",
                                   f"{length_ratio:.6f}", f"{width_ratio:.6f}"])
            if label_info:
                if z_value not in label_info_by_plane:
                    label_info_by_plane[z_value] = []
                label_info_by_plane[z_value].extend(label_info)
    else:
        raise ValueError("Unexpected data dimensions")

    return label_info_by_plane

def save_label_info(label_info_by_plane, save_dir, folder_name):
    for z_value, infos in label_info_by_plane.items():
        file_name = f"{folder_name}_{z_value}.txt"
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, 'a') as f:  # 使用 'a' 模式以追加内容
            for info in infos:
                f.write(' '.join(map(str, info)) + '\n')

def process_file(nrrd_path, save_label_dir):
    file_name = os.path.basename(nrrd_path)
    label_class = 0 if '_B' in file_name else 1
    folder_name = os.path.basename(os.path.dirname(nrrd_path))
    label_info_by_plane = process_nrrd_file(nrrd_path, label_class)
    save_label_info(label_info_by_plane, save_label_dir, folder_name)

def process_folder(folder_path, save_label_dir):
    nrrd_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nrrd'):
                nrrd_files.append(os.path.join(root, file))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, nrrd_path, save_label_dir) for nrrd_path in nrrd_files]
        for future in as_completed(futures):
            future.result()

def classify_txt_files(folder_path):
    for txt_file in os.listdir(folder_path):
        if txt_file == "classes.txt":
            continue
        patient_name = txt_file.split('_')[0]
        if not os.path.exists(f"{folder_path}/{patient_name}"):
            os.makedirs(f"{folder_path}/{patient_name}")
        new_file_name = f"{folder_path}/{patient_name}/{txt_file}"
        os.rename(f"{folder_path}/{txt_file}", new_file_name)


def copy_txt_file(output_folder: str):

    # 源文件路径
    source_file = r"G:\Program\DATABASE_TXT\classes.txt"

    # 目标文件夹路径
    target_folder = output_folder

    # 遍历目标文件夹中的所有子文件夹
    for root, dirs, files in os.walk(target_folder):
        for dir_name in dirs:
            # 构建子文件夹的完整路径
            subfolder_path = os.path.join(root, dir_name)

            # 构建目标文件的完整路径
            target_file = os.path.join(subfolder_path, os.path.basename(source_file))

            # 复制文件到子文件夹
            shutil.copy(source_file, target_file)

    print("文件已成功复制到所有子文件夹。")

# 设置文件夹路径(根目录即可，递归处理子文件夹)
input_folder = r"G:\Program\DATABASE\2020\Xiang\HuangChangKu"
output_label_folder = r"G:\Program\DATABASE_TXT"

# 创建保存文件夹
os.makedirs(output_label_folder, exist_ok=True)

# 处理文件夹
process_folder(input_folder, output_label_folder)

# classify_txt_files(output_label_folder)
# copy_txt_file(output_label_folder)