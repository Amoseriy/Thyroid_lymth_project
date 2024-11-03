#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/5/26 下午4:51
import os

import nibabel as nib
import numpy as np
import cv2

VEIN_IMAGE_TYPE = ['_ViDose', '_MonoE50keV', '_MonoE40keV', '_MonoE_V', '_V_', '_MonoE45keV', 'dataV_', '_vein_']

# 加载.nii.gz文件
def load_nii(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    return data


# 调整窗宽窗位
def apply_window(image, window_center, window_width):
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    adjusted_image = np.clip(image, img_min, img_max)
    adjusted_image = (adjusted_image - img_min) / (img_max - img_min)
    adjusted_image = (adjusted_image * 255).astype(np.uint8)
    return adjusted_image


def nii2jpg(nii_path, jpg_path, ww=360, wc=60):
    """
    :param nii_path: nii.gz文件路径，如"G:/Program/DATABASE\2023\Xiang\DuXueRu\DuXueRu_601_ViDose(3)_2mm_Chest normal.nii.gz"
    :param jpg_path: JPG文件保存的根目录"../DATABASE/JPG"
    :param ww: 窗宽：默认360
    :param wc: 窗位：默认60
    """
    nii_path = nii_path.replace('\\', '/').replace('\\\\', '/')
    ct_data = load_nii(nii_path)

    if not os.path.exists(jpg_path):
        os.makedirs(jpg_path)

    patient_name = nii_path.split('/')[-2]
    # research_name = nii_path.split("/")[-3]
    year = nii_path.split("/")[-4]
    # for-else结构：当for循环正常结束（即没有通过break退出循环）时，else块的代码会被执行。
    # 如果在循环中遇到break语句，则else块不会被执行。
    for vein_image_type in VEIN_IMAGE_TYPE:
        if vein_image_type in nii_path:
            series_name = vein_image_type.strip('_')
            break
    else:
        series_name = "have_not_found_vein_image_type"
    # print(research_name)
    adjusted_images = []
    for i in range(ct_data.shape[2]):
        ct_slice = ct_data[:, :, i]
        adjusted_image = apply_window(ct_slice, wc, ww)
        adjusted_image = cv2.rotate(adjusted_image, cv2.ROTATE_90_CLOCKWISE)
        adjusted_image = cv2.flip(adjusted_image, 1)  # 将图片水平翻转
        adjusted_images.append(adjusted_image)
        # print(adjusted_image.shape)

    # 保存调整后的图像为JPEG格式
    for i, image in enumerate(adjusted_images):
        # print(image)
        filename = f"{jpg_path}/{year}/{patient_name}"
        # 保存图片前先创建文件夹，不然无法保存
        if not os.path.exists(filename):
            os.makedirs(filename)
        filename = f"{filename}/{patient_name}_{i}.jpg"
        cv2.imwrite(filename, image)
        print(f"{filename} 已保存...")

    print("所有图像已转换并保存为JPEG格式。")


if __name__ == '__main__':
    # 加载.nii.gz文件
    nii_file = "../../DATABASE/2022/Ni/PanQiaoRong/PanQiaoRong_701_A+,iDose(3)_2mm_Chest normal.nii.gz"
    jpg_file = "./JPG/"
    nii2jpg(nii_file, jpg_file)
