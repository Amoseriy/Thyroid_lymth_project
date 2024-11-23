#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/8 19:02
import SimpleITK as sitk
import numpy as np
import os
import nrrd
import json

with open("./label_img_path_dict.json", "r") as f:
    label_img_path_dict = json.load(f)


root_dir = r'G:\Program\DATABASE'
output_dir = r'G:\Program\DATABASE_PNG'
for year in os.listdir(root_dir):
    year_dir = os.path.join(root_dir, year)
    for patient_name in os.listdir(year_dir):
        patient_dir = os.path.join(year_dir, patient_name)
        label_imgs_list = []
        # print(f"{'=' * 20} {patient_name} {'=' * 20}")
        for label_path in os.listdir(patient_dir):
            if label_path.endswith('.nrrd'):
                label_path = os.path.join(patient_dir, label_path)
                # print(f"Processing {label_path}")
                try:
                    img_path = label_img_path_dict[label_path]
                    print(f"{label_path} -> {img_path}")
                except KeyError:
                    print("=" * 50)
                    print(f"Warning: {label_path} not found in label_img_path_dict.json")
                    print("=" * 50)
                    continue


