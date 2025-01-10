#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/12/13 22:19
import os
import random
import shutil
from pathlib import Path

# 设置随机种子以保证结果可重复
random.seed(42)

# 定义源文件夹路径
source_images_dir = 'DATABASE_YOLO/image'
source_labels_dir = 'DATABASE_YOLO/label'

# 定义目标文件夹路径
target_train_dir = 'DATABASE_YOLO/image/train'
target_val_dir = 'DATABASE_YOLO/image/val'
target_test_dir = 'DATABASE_YOLO/image/test'
target_train_label_dir = 'DATABASE_YOLO/label/train'
target_val_label_dir = 'DATABASE_YOLO/label/val'
target_test_label_dir = 'DATABASE_YOLO/label/test'

# 创建目标文件夹
for dir in [target_train_dir, target_val_dir, target_test_dir,
            target_train_label_dir, target_val_label_dir, target_test_label_dir]:
    Path(dir).mkdir(parents=True, exist_ok=True)

# 获取所有图像文件名（假设标签文件名与图像文件名相同，只是扩展名不同）
image_filenames = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 随机打乱文件名列表
random.shuffle(image_filenames)

# 计算每个集合的数量
total_files = len(image_filenames)
train_split = int(0.7 * total_files)
val_split = int(0.85 * total_files)  # 70% + 15%

# 分割文件名列表
train_files = image_filenames[:train_split]
val_files = image_filenames[train_split:val_split]
test_files = image_filenames[val_split:]


# 定义复制函数
def copy_files_to_target(file_list, target_image_dir, target_label_dir):
    for filename in file_list:
        base_name, ext = os.path.splitext(filename)
        label_filename = base_name + '.txt'  # 假设标签文件为txt格式

        # 复制图像文件
        shutil.move(os.path.join(source_images_dir, filename), target_image_dir)
        # 复制标签文件
        shutil.move(os.path.join(source_labels_dir, label_filename), target_label_dir)


# 将文件复制到相应的目标文件夹
copy_files_to_target(train_files, target_train_dir, target_train_label_dir)
copy_files_to_target(val_files, target_val_dir, target_val_label_dir)
copy_files_to_target(test_files, target_test_dir, target_test_label_dir)