#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/10 下午8:45
import os
import concurrent.futures
from bin.nii2jpg import nii2jpg
import json


def process_file(file_name,  output_base_path):
    """
    定义在每个线程中执行的函数，用以处理单个文件
    """
    file_path = file_name
    # patient_name = file_name.split("_")[0]
    # output_path = os.path.join(output_base_path, patient_name)
    os.makedirs(output_base_path, exist_ok=True)
    nii2jpg(file_path, output_base_path)
    print(f"Processed {file_name}")


def main():
    # with open("../error_path.json", "r") as f:
    #     path_dict = json.load(f)
    output_base_path = "./DATABASE_JPG"
    # file_names = path_dict.values()
    file_names = [
"G:\Program\DATABASE\\2020\Xiang\ChenZhiQiang\ChenZhiQiang_602_MonoE45keVlocatorSpectral(3)_1mm_Chest normal.nii.gz",
"G:\Program\DATABASE\\2020\Xiang\GuoYiFan\GuoYiFan_602_MonoE45keVlocatorSpectral(3)_1mm_Chest normal.nii.gz",
"G:\Program\DATABASE\\2020\Xiang\HongJianMin\HongJianMin_702_MonoE45keVlocatorSpectral(3)_1mm_Chest normal.nii.gz",
"G:\Program\DATABASE\\2020\\Ni\SongQiuQin\\602_MonoE45keVlocatorSpectral(3)_1mm_Chest normal.nii.gz",
"G:\Program\DATABASE\\2020\\Ni\SongWeiNa\\603_MonoE50keVVSpectral(4)_1mm_Neck C 3X3.nii.gz",
"G:\Program\DATABASE\\2022\Gao\HongShangYou\HongShangYou_705_MonoE40keV,V+,Spectral(4)_1mm_Chest normal.nii.gz",
"G:\Program\DATABASE\\2023\Gao\WangLiYue\WangLiYue_MonoE_V_image.nii.gz"
]
    # for file_name in file_names:
    #     process_file(file_name, output_base_path)

    # 创建线程池，最大并发线程数为5
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # 提交所有任务到线程池，future保存所有提交的任务对象
        futures = [executor.submit(process_file, file_name, output_base_path) for file_name in file_names]

        # 等待所有线程完成
        # concurrent.futures.as_completed：返回一个迭代器，按任务完成的顺序产生 Future 对象。
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # 获取线程运行结果，如果有异常会在这里抛出
            except Exception as e:
                print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
