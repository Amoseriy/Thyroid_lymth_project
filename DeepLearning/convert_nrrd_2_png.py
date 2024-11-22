#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/22 15:06
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/22 15:06
import os
import SimpleITK as sitk
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pathlib

# 读取路径字典
with open(r'G:\Program\label_img_path_dict.json', 'r') as f:
    rect_path_dict = json.load(f)

require_resample = {}
new_path_dict = {}

# 定义处理单个图像的函数
def process_image(key, value):
    try:
        img = sitk.ReadImage(value)
        label = sitk.ReadImage(key)
    except Exception as e:
        print(f"{'=' * 20}")
        print(f"Error reading image {key} or label {value}: {e}")
        print(f"{'=' * 20}")
        return None
    if img.GetSize() != label.GetSize():
        print(f"Image {key} and label {value} have different size")
        return key, value
    return None


# 使用线程池执行多线程处理
with ThreadPoolExecutor() as executor:
    futures = []
    for key,value in rect_path_dict.items():

        futures.append(executor.submit(process_image, key,value))

    for future in as_completed(futures):
        result = future.result()
        if result:
            key, value = result
            require_resample[key] = value



# 保存结果
with open(r'G:\Program\require_resample.json', 'w') as f:
    json.dump(new_path_dict, f)


