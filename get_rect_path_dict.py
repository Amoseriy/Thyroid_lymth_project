#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/8 17:24
import os
import json

from skimage.color import bro_from_rgb

root1 = "./DATABASE_RECT"
root2 = "./DATABASE"
VEIN_IMAGE_TYPE = ['_ViDose', '_MonoE50keVV', '_MonoE40keVV', '_MonoE_V', '_V_',
                   '_MonoE45keVV', 'dataV_', '_vein_', '_V+,iDose', '_NeckViDose', '_MonoE40keV_V_']

nii_dict = {}
last_list = []
for root2, dirs2, files2 in os.walk(root2):
    if not dirs2:
        patient_id = os.path.basename(root2)
        # last_list.append(patient_id)
        tmp_list = [temp for temp in os.listdir(root2) if temp.endswith(".nii.gz")]
        tmp_list = [os.path.join(root2, temp) for temp in tmp_list]
        nii_dict[patient_id] = tmp_list

# print(nii_dict)

final_dict = {}
for root1, dirs1, files1 in os.walk(root1):
    for file1 in files1:
        if file1.endswith(".nrrd"):
            file_path1 = os.path.join(root1, file1)
            # print(file_path1)
            patient_id = file1.split("_")[0]
            # print(patient_id)
            nii_list = nii_dict.get(patient_id)
            # print(nii_list)
            tem_list = []
            for nii_file in nii_list:
                if  any(char in nii_file for char in VEIN_IMAGE_TYPE):
                    tem_list.append(nii_file)
            if len(tem_list) == 0:
                print("No vein image found for patient: ", patient_id)
            elif len(tem_list) == 1:
                final_dict[file_path1] = tem_list[0]
            elif len(tem_list) > 1:
                for file_name in tem_list:
                    if "_MonoE_V_" in file_name:
                        final_dict[file_path1] = file_name
                        break
                    elif "Spectral(4)_1mm" in file_name:
                        final_dict[file_path1] = file_name
                        break
                    elif "MonoE45Spectral" in file_name:
                        final_dict[file_path1] = file_name
                        break
                    elif "Spectral(3)_1mm" in file_name:
                        final_dict[file_path1] = file_name
                        break
                    elif "MonoE50Spectral(0)_1mm" in file_name:
                        final_dict[file_path1] = file_name
                        break
                    elif "_MonoE40keV_V_" in file_name:
                        final_dict[file_path1] = file_name
                        break
                    else:
                        print(file_name)
            print("-" * 50)

# print(final_dict)

with open("rect_path_dict.json", "w") as f:
    json.dump(final_dict, f)