#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/3 15:41
import os
import re
import shutil
import json


root_jpg = "./DATABASE_JPG"
root_txt = "./DATABASE_TXT"

def get_jpg_dict(root_path):
    leaf_dir = []
    for root, dirs, files in os.walk(root_path):
        # print(dirs)
        if len(dirs) == 0:
            leaf_dir.append(root)

    return_dict = {}
    for leaf in leaf_dir:
        num_list = []
        for file in os.listdir(leaf):
            if file == "classes.txt":
                continue
            pattern = re.compile(r'\d+')
            num = pattern.findall(file)
            num_list.append(int(num[0]))
        num_list.sort()
        # print(num_list)
        return_dict[leaf.split("\\")[-1]] = max(num_list)
    return return_dict


def get_patient_dict(root_path, patient_list):
    patient_dict = {}
    for root, dirs, files in os.walk(root_path):
        if len(dirs) == 0:
            patient_name = root.split("\\")[-1]
            if patient_name in patient_list:
                patient_dict[patient_name] = root
    return patient_dict


if __name__ == '__main__':
    jpg_dict = get_jpg_dict(root_jpg)
    # print(jpg_dict)
    txt_dict = get_jpg_dict(root_txt)
    # print(txt_dict)

    key_list = []
    temp_dict = {}
    for key in jpg_dict.keys():
        jpg_num = jpg_dict[key]
        txt_num = txt_dict[key]
        if jpg_num < txt_num:
            print(f"{key} has {jpg_num} jpg files, but {txt_num} txt files.")
            key_list.append(key)
            # temp_dict[key] = txt_num
    # with open("error_file.json", "w") as f:
    #     json.dump(temp_dict, f)
    # target_dict = get_patient_dict("./DATABASE", key_list)
    # print(target_dict)


