#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/10/22 21:41
import numpy as np
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
import json
from bin import resample_file
# Pool用于创建进程池，Manager用于创建进程间共享的数据结构，管理进程间通信
from multiprocessing import Pool, Manager
from tqdm import tqdm


# 读取文件字典
def get_file_dict():
    with open("../rect_path_dict.json", "r") as f:
        label_dict = json.load(f)
    return label_dict


# 获取标签值
def get_label_numbers(lab_path):
    label = sitk.ReadImage(lab_path)
    label_array = sitk.GetArrayFromImage(label)
    unique_labels = np.unique(label_array)
    return unique_labels


# 特征提取函数
def extract_features(item, file_dict, param_path):
    try:
        label_path = item.replace("\\", "/")
        img_path = file_dict[item]
        print(f"====================开始提取:{item}====================")
        # print(label_path)

        patient_id = label_path.split("/")[-2]
        temp = label_path.split("_")
        target = f"_{temp[-2]}_{temp[-1][0]}"

        lab_num = get_label_numbers(label_path, img_path)

        label = sitk.ReadImage(label_path)
        image = sitk.ReadImage(img_path)
        print("------------------------------------------------")
        print(f"标签文件size为:{label.GetSize()}")
        print(f"图像文件size为:{image.GetSize()}")
        if label.GetSize() != image.GetSize():
            label = resample_file(image, label)
            print(f"重采样后的label文件size为:{label.GetSize()}")

        extractor = featureextractor.RadiomicsFeatureExtractor(param_path)
        results = []

        for label_value in lab_num[1:]:
            extractor.settings['label'] = label_value
            print(f"1.标签 {label_value} 设置完毕！")
            result = extractor.execute(image, label)
            print(f"2.标签 {label_value} 提取完毕！")
            result_series = pd.Series(result)
            value = patient_id + "_" + str(label_value) + target
            result_series['Label'] = value
            results.append(result_series)
        # 一个label文件提取完毕后，将结果合并到一个DataFrame中
        # 然后传递给进程的回调函数
        return pd.concat(results, axis=1)

    except Exception as e:
        print(e)
        # 如果出现异常，则返回None，回调函数会忽略这个结果，不然回调函数将不会被调用
        return None


def main():
    param_path = "Params.yaml"
    file_dict = get_file_dict()

    with Manager() as manager:
        # 创建一个与普通Python列表类似的共享列表。所有进程都可以把结果添加到这个列表中。
        results = manager.list()
        with Pool(processes=4) as pool:  # 设置进程数
            for item in file_dict.keys():  # item为label_path
                # apply_async：允许我们异步地提交一个任务到进程池。它不会阻塞主线程，允许同时提交多个任务。
                # callback：当任务完成时，会调用回调函数。此处的回调函数将extract_features的返回值添加到共享列表中。
                # 回调函数一般用于处理进程池的结果，比如保存到文件或数据库。
                result = pool.apply_async(extract_features, (item, file_dict, param_path),
                                          callback=results.append)

            pool.close()  # 在所有的任务都已提交后，关闭进程池，不再接受新的任务，但已提交的任务会继续执行。
            pool.join()  # 在with关闭进程池之前调用join，这个调用会阻塞主进程，直到所有进程都完成，保证任务执行的完整性调用之前必须先调用close()，。

        # 所有线程的结果都保存在results列表中
        features_df = pd.concat([df for df in results if df is not None], axis=1)

        # 设置索引为标签值
        features_df = features_df.T
        features_df.set_index('Label', inplace=True)

        # 保存到 CSV
        csv_file_path = './extracted_features_rect.csv'
        features_df.to_csv(csv_file_path)


if __name__ == '__main__':
    main()
