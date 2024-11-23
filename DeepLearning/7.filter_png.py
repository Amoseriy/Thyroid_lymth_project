#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/22 23:40
import os

import torch
from PIL import Image
from torchvision.io import decode_image
import numpy as np
import shutil
from pathlib import Path


root_dir = r"G:\Program\DATABASE_PNG"

target_file_list = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.png'):
            mask_path = os.path.join(root, file)
            print(mask_path)
            mask_tensor = decode_image(mask_path)
            # print(mask_tensor.shape)
            obj_ids = torch.unique(mask_tensor)
            print(obj_ids)
            if len(obj_ids) > 1:
                target_file_list.append(mask_path)

for ori_path in target_file_list:
    # print(ori_path)
    new_path = ori_path.replace("_PNG", "_PNG2")
    print(f"{ori_path} -> {new_path}")
    path = Path(new_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    shutil.copy(ori_path, new_path)


