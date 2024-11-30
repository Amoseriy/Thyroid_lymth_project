#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/22 15:06
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/22 15:06
import os
import SimpleITK as sitk
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from bin import resample_file
import nrrd
from skimage.measure import label as ski_label


# 读取路径字典
with open(r'G:\Program\label_img_path_dict.json', 'r') as f:
    label_img_path_dict = json.load(f)

# with open(r"G:\Program\require_resample.json","r") as f:
#     resample_dict = json.load(f)

def get_label_img(label_path, img_path):
    # print(f"Processing {label_path}")
    img_size = sitk.ReadImage(img_path)
    label_size = sitk.ReadImage(label_path)
    # print(f"{img_size.GetSize() =} \n {label_size.GetSize() =}")
    # if label_size.GetSize() != img_size.GetSize():
    label = resample_file(img_size, label_size)
        # label = sitk.Cast(label, sitk.sitkUInt8)  # 转换为uint8类型, 便于保存为png格式
    # else:
    #     label = label_size
   # 将SimpleITK图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(label)
    # print(image_array.shape)
    if len(image_array.shape) == 4:
        # 创建每个通道的arr字典，以便后来合并
        image_array_dict = {}
        for t in range(image_array.shape[3]):
            t_slice_image = image_array[:, :, :, t]
            # 保存切片图像
            image_array_list = []
            for i in range(image_array.shape[0]):
                slice_image = t_slice_image[i, :, :]
                # print(slice_image.shape)
                image_array_list.append(slice_image)
            image_array_dict[t] = image_array_list
        # print(image_array_dict)
        # 获取字典中的所有矩阵列表
        matrix_lists = list(image_array_dict.values())
        # 使用 zip 函数将每个位置上的矩阵配对
        combined_masks = []
        for matrices in zip(*matrix_lists):
            # 初始化一个与矩阵相同形状的全零数组
            combined_mask = np.zeros_like(matrices[0])
            # 遍历每个矩阵并更新 combined_mask
            for matrix in matrices:
                # 使用逻辑或操作更新 combined_mask
                combined_mask = np.where((combined_mask == 0) & (matrix != 0), matrix, combined_mask)
            # 将合并后的掩膜添加到结果列表中, 形状为（n, 512, 512）
            combined_masks.append(combined_mask)
        # print(combined_masks)
        combined_masks = np.stack(combined_masks, axis=0)
        # 返回label数组
        slice_image_itk = combined_masks # (512, 512, 194)
        return slice_image_itk

    elif len(image_array.shape) == 3:
        # 返回label数组
        slice_image_itk = image_array # (512, 512, 194)
        return slice_image_itk


def save_label_img(label_imgs_list, output_dir, year, patient_name, classify):
    if len(label_imgs_list) == 0:
        return None
    # 将列表转换为NumPy数组堆栈
    mask_stack = np.stack(label_imgs_list, axis=0)
    # 按照第一个新创建的维度（这里是指第0个维度）求和
    sum_masks = np.sum(mask_stack, axis=0)
    # print(f"{sum_masks.shape = }")

    slice_image_itk = sitk.GetImageFromArray(sum_masks)
    slice_image_itk = sitk.Cast(slice_image_itk, sitk.sitkUInt8)  # 转换为uint8类型, 便于保存为png格式
    # print(f"{slice_image_itk.GetSize() = }")
    for i in range(slice_image_itk.GetSize()[2]):
        if classify == "benign":
            output_file_path = Path(output_dir) / f'{year}/{patient_name}/benign/{patient_name}_{i}.png'
        else:
            output_file_path = Path(output_dir) / f'{year}/{patient_name}/malignant/{patient_name}_{i}.png'
        if not output_file_path.parent.exists():
            output_file_path.parent.mkdir(parents=True)
        # print(output_file_path)
        sitk.WriteImage(slice_image_itk[:, :, i], str(output_file_path))
        print(f"Saved {output_file_path}")


if __name__ == '__main__':
    root_dir = r'G:\Program\DATABASE'
    output_dir = r'G:\Program\DATABASE_PNG'
    for year in os.listdir(root_dir):
        year_dir = os.path.join(root_dir, year)
        for patient_name in os.listdir(year_dir):
            patient_dir = os.path.join(year_dir, patient_name)
            label_imgs_list_benign = []
            label_imgs_list_malignant = []
            print(f"{'=' * 20} {patient_name} {'=' * 20}")
            for label_path in os.listdir(patient_dir):
                if label_path.endswith('.nrrd'):
                    label_path = os.path.join(patient_dir, label_path)
                    print(f"Processing {label_path}")
                    img_path = label_img_path_dict[label_path]
                    label_img = get_label_img(label_path, img_path)
                    if "_B.seg.nrrd" in label_path:
                        label_imgs_list_benign.append(label_img)
                    else:
                        label_imgs_list_malignant.append(label_img)
                    # print(f"Processing {label_path}")
            save_label_img(label_imgs_list_benign, output_dir, year, patient_name, "benign")
            save_label_img(label_imgs_list_malignant, output_dir, year, patient_name, "malignant")





