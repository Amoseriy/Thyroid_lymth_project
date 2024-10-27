#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/10/22 21:41
from pathlib import Path
import numpy as np
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
import json
from bin import resample_file


def get_file_dict():
    with open("../path_dict.json", "r") as f:
        label_dict = json.load(f)

    return label_dict


def get_label_numbers(lab_path, img_path):
    # 读取标签图像
    label = sitk.ReadImage(lab_path)
    # 将 SimpleITK 图像转换为 NumPy 数组
    label_array = sitk.GetArrayFromImage(label)
    # 获取唯一标签值
    unique_labels = np.unique(label_array)
    return unique_labels


def main():
    param_path = "Params.yaml"
    file_dict = get_file_dict()
    extractor = featureextractor.RadiomicsFeatureExtractor(param_path)
    features_df = pd.DataFrame()
    error_list = []
    for item in file_dict.keys():
        try:
            # print(item)
            label_path = item
            img_path = file_dict[item]
            print(f"====================开始提取:{item}====================")
            patient_id = label_path.split("\\")[-2]
            # 从文件名中提取label列的部分值，用于构造label列
            temp = label_path.split("_")
            target = f"_{temp[-2]}_{temp[-1][0]}"

            lab_num = get_label_numbers(label_path, img_path)

            label = sitk.ReadImage(label_path)
            image = sitk.ReadImage(img_path)
            print("------------------------------------------------")
            print(f"标签文件size为:{label.GetSize()}")
            print(f"图像文件size为:{image.GetSize()}")
            if label.GetSize() != image.GetSize():
                label = resample_file(image, label)
                # 现在，resampled_mask的尺寸应该与original_image一致
                print(f"重采样后的label文件size为:{label.GetSize()}")
            for label_value in lab_num[1:]:
                # 设置当前的标签值
                extractor.settings['label'] = label_value
                print(f"1.标签 {label_value} 设置完毕！")
                # 提取特征
                result = extractor.execute(image, label)
                # 输出结果，或者根据需要保存结果
                print(f"2.标签 {label_value} 提取完毕！")
                # 将结果转换为 pandas Series（确保结果中的第一个元素是标签值）
                result_series = pd.Series(result)
                value = patient_id + "_" + str(label_value) + target
                result_series['Label'] = value  # 添加标签值作为一个列
                # print(result_series)
                # 将这个系列添加到数据帧中
                features_df = pd.concat([features_df, result_series], axis=1)
                print("------------------------------------------------")
                # print(features_df)
                # print(f"5.标签 {label_value} 提取完毕！")
            print(f"===================={item}提取完毕！====================")
        except Exception as e:
            # error_list.append(str(value))
            print(e)
            continue

    # 设置索引为标签值
    features_df = features_df.T
    features_df.set_index('Label', inplace=True)
    # 保存到 CSV
    csv_file_path = './extracted_features.csv'
    features_df.to_csv(csv_file_path)
    # print(error_list)


if __name__ == '__main__':
    main()
