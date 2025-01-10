#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/8 19:02
import os
import random
import shutil
from math import ceil

def split_data(source, training, validation, testing, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 创建目标文件夹
    for folder in [training, validation, testing]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        for class_dir in ['benign', 'malignant']:
            class_path = os.path.join(folder, class_dir)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

    # 遍历源文件夹中的每个类别
    for class_dir in ['benign', 'malignant']:
        source_class_dir = os.path.join(source, class_dir)
        files = os.listdir(source_class_dir)
        random.shuffle(files)  # 打乱文件顺序以确保随机性

        total_files = len(files)
        num_train = ceil(total_files * train_ratio)
        num_val = ceil(total_files * val_ratio)

        # 将文件移动到对应的训练集、验证集和测试集中
        for i, file_name in enumerate(files):
            if i < num_train:
                dest_folder = training
            elif i < num_train + num_val:
                dest_folder = validation
            else:
                dest_folder = testing

            src_file = os.path.join(source_class_dir, file_name)
            dest_file = os.path.join(dest_folder, 'benign' if class_dir == 'benign' else 'malignant', file_name)
            shutil.move(src_file, dest_file)  # 使用shutil.move()如果想移动而不是复制文件

source_folder = './DATABASE_YOLO_CLS'
training_folder = 'train'
validation_folder = 'val'
testing_folder = 'test'

split_data(source_folder, training_folder, validation_folder, testing_folder)
