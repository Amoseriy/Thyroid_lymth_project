#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/10 下午4:50

import os
import nrrd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_nrrd_file(nrrd_path, label_class):
    data, _ = nrrd.read(nrrd_path)
    label_info_by_plane = {'Axial': {}}
    unique_labels = np.unique(data)

    # 确定数据的维度
    if data.ndim == 4:
        z_slices = data.shape[3]
    elif data.ndim == 3:
        z_slices = data.shape[2]
    else:
        raise ValueError("Unexpected data dimensions")

    # 处理轴向切片
    unique_z_values = np.unique(np.where(data > 0)[-1])
    unique_z_values = unique_z_values[unique_z_values < z_slices]
    for z_value in unique_z_values:
        label_info = []
        z_slice_data = data[:, :, z_value] if data.ndim == 3 else data[:, :, :, z_value]
        for label in unique_labels:
            if label != 0:
                label_coords = np.where(z_slice_data == label)
                if label_coords[0].size > 0:
                    min_x, max_x = min(label_coords[0]), max(label_coords[0])
                    min_y, max_y = min(label_coords[1]), max(label_coords[1])
                    length, width = max_x - min_x + 1, max_y - min_y + 1
                    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
                    center_x_ratio = center_x / data.shape[0]
                    center_y_ratio = center_y / data.shape[1]
                    length_ratio = length / data.shape[0]
                    width_ratio = width / data.shape[1]
                    label_info.append([label_class, f"{center_x_ratio:.6f}", f"{center_y_ratio:.6f}",
                                       f"{length_ratio:.6f}", f"{width_ratio:.6f}"])
        label_info_by_plane['Axial'][z_value] = label_info

    return label_info_by_plane

def save_label_info(label_info_by_plane, save_dir, folder_name):
    for plane, label_info_by_z in label_info_by_plane.items():
        for z_value, infos in label_info_by_z.items():
            file_name = f"{folder_name}_{z_value}.txt"
            file_path = os.path.join(save_dir, file_name)
            with open(file_path, 'w') as f:
                for info in infos:
                    f.write(' '.join(map(str, info)) + '\n')
                    print(f"Saved {file_path}")

def process_file(nrrd_path, save_label_dir):
    label_class = 0 if '_B.' in nrrd_path else 1
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
        patient_name = txt_file.split('_')[0]
        if not os.path.exists(f"{folder_path}/{patient_name}"):
            os.makedirs(f"{folder_path}/{patient_name}")
        new_file_name = f"{folder_path}/{patient_name}/{txt_file}"
        os.rename(f"{folder_path}/{txt_file}", new_file_name)

# 设置文件夹路径(根目录即可，递归处理子文件夹)
input_folder = r"G:\Program\DATABASE\2022\Xiang\CaiXueYing"
output_label_folder = r"./DATABASE_TXT"

# 创建保存文件夹
os.makedirs(output_label_folder, exist_ok=True)

# 处理文件夹
process_folder(input_folder, output_label_folder)
classify_txt_files(output_label_folder)