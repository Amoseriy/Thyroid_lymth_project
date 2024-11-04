#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/4 19:54
import os
import shutil

root_path1 = "./DATABASE_JPG"
root_path2 = "./DATABASE_TXT"

DICT = {}
for year in os.listdir(root_path1):
    year_path = os.path.join(root_path1, year)
    for patient_name in os.listdir(year_path):
        DICT[patient_name] = year

print(DICT)

for name in os.listdir(root_path2):
    if name in DICT:
        year = DICT[name]
        year_path = os.path.join(root_path2, year)
        if not os.path.exists(year_path):
            os.makedirs(year_path)
        new_path = os.path.join(year_path, name)
        print(f"{os.path.join(root_path2, name)} -> {new_path}")
        shutil.move(os.path.join(root_path2, name), new_path)