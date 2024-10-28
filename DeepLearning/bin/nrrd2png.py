#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/16 上午11:42
import SimpleITK as sitk
import numpy as np
import imageio
import os


def nrrd2png(mask_nrrd_file, output_dir):
    file = mask_nrrd_file.split("/")[-1]
    patient_name = file.split("_")[0]
    quality = f"_{file.split('_')[-2][0]}_{file.split('_')[-1][0]}"
    patient_name = f"{patient_name}{quality}"
    # patient_name = file[:len(patient_name) + 4]

    researcher_name = mask_nrrd_file.split("/")[-3]
    # 读取nrrd格式的掩膜文件和CT文件
    mask_image = sitk.ReadImage(mask_nrrd_file)
    # ct_image = sitk.ReadImage(ct_nrrd_file)

    # 获取掩膜文件和CT文件的数组
    mask_array = sitk.GetArrayFromImage(mask_image)
    # ct_array = sitk.GetArrayFromImage(ct_image)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 确保掩膜文件和CT文件的切片数量一致
    # assert mask_array.shape[0] == ct_array.shape[0], "掩膜文件和CT文件的切片数量不一致"

    # 遍历每一个切片并保存为png格式
    for i in range(mask_array.shape[0]):
        # 删除第一张和最后一张
        if i == 0 or i == mask_array.shape[0] - 1:
            continue
        mask_slice = mask_array[i, :, :]
        # ct_slice = ct_array[i, :, :]

        # 保存掩膜切片为png文件
        mask_output_path = os.path.join(output_dir, f"{researcher_name}_{patient_name}_{i}.png")
        imageio.imwrite(mask_output_path, mask_slice.astype(np.uint8))

        # 保存CT切片为png文件（可选，如果需要的话）
        # ct_output_path = os.path.join(output_dir, f"ct_{i}.png")
        # imageio.imwrite(ct_output_path, ct_slice.astype(np.uint8))

    print(f"转换完成，共保存了 {mask_array.shape[0] - 2} 个切片")


if __name__ == "__main__":
    # 示例用法
    mask_nrrd_file = '../../Data/Xiang/Label_NRRD/BaiMaoAn_L_B-label.nrrd'
    # ct_nrrd_file = 'path/to/ct_file.nrrd'
    output_dir = '../PNG/'
    nrrd2png(mask_nrrd_file, output_dir)
