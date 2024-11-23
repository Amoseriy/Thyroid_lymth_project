#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/7/3 下午10:04
import SimpleITK as sitk
import numpy as np


def resample_file(target_file, resample_file):
    # 获取原始图像的尺寸、原点、方向和像素类型
    print(f"开始重采样!!!")
    output_size = target_file.GetSize()
    output_spacing = target_file.GetSpacing()
    output_origin = target_file.GetOrigin()
    output_direction = target_file.GetDirection()

    # 创建重采样过滤器
    resample_filter = sitk.ResampleImageFilter()

    # 设置重采样过滤器的参数
    resample_filter.SetSize(output_size)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetOutputDirection(output_direction)
    resample_filter.SetDefaultPixelValue(0)

    # 设置插值方法，对于掩膜图像通常使用最近邻插值
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)

    # 执行掩膜的重采样
    resampled = resample_filter.Execute(resample_file)
    print(f"重采样完成!!!")
    return resampled


def get_label_numbers(lab_path):
    # 读取标签图像
    label = sitk.ReadImage(lab_path)
    # 将 SimpleITK 图像转换为 NumPy 数组
    label_array = sitk.GetArrayFromImage(label)
    # 获取唯一标签值
    unique_labels = np.unique(label_array)
    return unique_labels


if __name__ == '__main__':
    # 原始图像路径
    target_path = r"G:/Program/DATABASE\\2020\\SongQiuQin\\701_ViDose(3)_2mm_Chest normal.nii.gz"
    # 标签图像路径
    lab_path = r"G:/Program/DATABASE\\2020\\SongQiuQin\\SongQiuQin_MonoE_LLLN_M.seg.nrrd"

    target_file = sitk.ReadImage(target_path)
    lab_file = sitk.ReadImage(lab_path)

    print(f"{target_file.GetSize()=}, {lab_file.GetSize()=}")
    resampled_file = resample_file(target_file, lab_file)
    print(f"{resampled_file.GetSize()=}")

