#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/10 下午9:12
import cv2
import os

"""
获取jpg目标区域截图
"""

def process_jpg(file_path):
    patient_num = file_path.split('/')[-1][:-4]
    # print(patient_name)
    # 读取文本文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析每行的标签和坐标
    annotations = []
    for line in lines:
        parts = line.strip().split()
        label = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        annotations.append((label, x_center, y_center, width, height))

    # 假设原始图像为 image.jpg
    image_path = file_path.replace("TXT", "JPG").replace(".txt", ".jpg")
    print(image_path)
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # 创建输出目录
    os.makedirs("../DATABASE_DL_JPG/benign/", exist_ok=True)
    os.makedirs("../DATABASE_DL_JPG/malignant/", exist_ok=True)

    # 截取并保存图片
    for i, (label, x_center, y_center, width, height) in enumerate(annotations):
        # 转换为图像上的像素坐标
        x_center_pixel = int(x_center * img_width)
        y_center_pixel = int(y_center * img_height)
        width_pixel = int(width * img_width)
        height_pixel = int(height * img_height)

        # 计算左上角和右下角的坐标
        x1 = max(0, x_center_pixel - width_pixel // 2 - 2)
        y1 = max(0, y_center_pixel - height_pixel // 2 - 2)
        x2 = min(img_width, x_center_pixel + width_pixel // 2 + 2)
        y2 = min(img_height, y_center_pixel + height_pixel // 2 + 2)

        # 截取图像区域
        cropped_image = image[y1:y2, x1:x2]
        # print(x1, y1, x2, y2)

        # 保存截取的图片
        if label == 0:
            output_path = f"../DATABASE_DL_JPG/benign/{patient_num}_{label}_{i + 1}.jpg"
            print(f"{output_path} 已写入。。。")
            cv2.imwrite(output_path, cropped_image)
        else:
            output_path = f"../DATABASE_DL_JPG/malignant/{patient_num}_{label}_{i + 1}.jpg"
            print(f"{output_path} 已写入。。。")
            cv2.imwrite(output_path, cropped_image)


def main():
    ori_txt_path = "../DATABASE_TXT"
    ori_jpg_path = "../DATABASE_JPG"
    for year in os.listdir(ori_txt_path):
        year_path = os.path.join(ori_txt_path, year)
        if year == "classes.txt":
            continue
        for name in os.listdir(year_path):
            if name == "classes.txt":
                continue
            # print(txt)
            txt_path = os.path.join(year_path, name)
            for txt in os.listdir(txt_path):
                if txt == "classes.txt":
                    continue
                target_txt_path = os.path.join(txt_path, txt).replace("\\\\", "/")
                # print(target_txt_path)
                process_jpg(target_txt_path)


if __name__ == "__main__":
    main()
