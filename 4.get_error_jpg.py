#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/3 17:15
import SimpleITK as sitk
import os
import nrrd
import json


with open('error_jpg_path.json', 'r') as f:
    error_jpg_dict = json.load(f)

with open('error_txt_num.json', 'r') as f:
    error_txt_num_dict = json.load(f)

def get_error_path():
    path_dict = {}
    for jpg_path in error_jpg_dict.values():
        temp_list = []
        patients_id = jpg_path.split('\\')[-1]
        # if patients_id == "CaiXueYing":
        #     for file in os.listdir(jpg_path):
        #         img_path = os.path.join(jpg_path, file)
        #         data = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        #         print(f"{img_path} 的维度为： {data.shape}")
        print(f"开始检查{patients_id}的nii.gz文件")
        for file in os.listdir(jpg_path):

            if file.endswith('nii.gz'):
                img_path = os.path.join(jpg_path, file)
                txt_num = error_txt_num_dict[patients_id]
                data = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                if data.shape[0] >= txt_num:
                    print(f"The txt number of {patients_id} is {txt_num}")
                    print(f"The shape of {img_path} is {data.shape[0]}")
                    temp_list.append(img_path)
                # print(temp_list)
        path_dict[patients_id] = temp_list
        print("-" * 20)

    with open("error_path.json", "w") as f:
        json.dump(path_dict, f)

if __name__ == '__main__':
    with open('error_path.json', 'r') as f:
        error_path_dict = json.load(f)
    temp_dict1 = {}
    temp_dict2 = {}
    for key, value in error_path_dict.items():
        if len(value) > 1:
            temp_dict1[key] = value
        else:
            temp_dict2[key] = value
    temp_dict3 = {}
    for key, value in temp_dict1.items():
        temp_dict3[key] = [x for x in value if "_A_" not in x]
    print(temp_dict3)
    # print(temp_dict)


