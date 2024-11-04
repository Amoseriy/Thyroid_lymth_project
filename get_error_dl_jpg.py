#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/3 23:10
from PIL import Image
import os


def check_image_ratio(image_path):
    try:
        # 打开图片文件
        with Image.open(image_path) as img:
            # 获取图片的宽度和高度
            width, height = img.size

            # 检查长宽比是否满足条件
            if height > 3 * width or width > 3 * height:
                return os.path.basename(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    return None


# 示例用法
root = "./DATABASE_DL_JPG"
name_list = []
for root, dirs, files in os.walk(root):
    for file in files:
        if file.endswith(".jpg"):
            file_path = os.path.join(root, file)
            result = check_image_ratio(file_path)
            if result:
                name = result.split("_")[0]
                name_list.append(name)
                print(f"The file '{result}' has a length-to-width ratio greater than 3.")

name_list = list(set(name_list))
print(name_list)
print(len(name_list))