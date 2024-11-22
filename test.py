#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/8 19:02
import SimpleITK as sitk
import numpy as np
import os
import nrrd

# 设置要读取的目录路径
directory_path = r"G:/Program/DATABASE"

# 遍历目录及其子目录中的所有文件
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.seg.nrrd'):
            file_path = os.path.join(root, file)
            # print(f"Processing file: {file_path}")
            dim = sitk.ReadImage(file_path).GetDimension()
            if dim == 4:
                print(f"Processing file: {file_path}")
            else:
                continue



