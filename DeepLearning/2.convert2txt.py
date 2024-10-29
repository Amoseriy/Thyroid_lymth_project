#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/10 下午4:50
import os
from bin.nrrd_to_txt import process_folder
import shutil
import json


with open("../path_dict.json", "r") as f:
    path_dict = json.load(f)

output_path = "G:/Program/DATABASE_TXT"
os.makedirs(output_path, exist_ok=True)
for ori_path in path_dict.keys():
    process_folder(ori_path, output_path)

file_names = os.listdir(output_path)

for file_name in file_names:
    patient_name = "".join([char for char in file_name.split(".")[0] if not char.isdigit()])
    # print(patient_name)
    if patient_name == "classes":
        continue
    dest_path = os.path.join(output_path, patient_name)
    # print(dest_path)
    os.makedirs(dest_path, exist_ok=True)

    os.rename(os.path.join(output_path, file_name), os.path.join(dest_path, file_name))
    shutil.copy2(os.path.join(output_path, "classes.txt"), os.path.join(dest_path, "classes.txt"))
