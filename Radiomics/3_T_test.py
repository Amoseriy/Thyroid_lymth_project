#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/2/20 18:25
"""
用python做T检验,筛选特征
"""
import pandas as pd
# levene检验方差是否相等, ttest_ind进行独立样本t检验
from scipy.stats import levene, ttest_ind

tData = pd.read_csv('./data/rect_FilteredTotalData.csv', low_memory=False)
# 确保数据中只有数值类型
for column_name in tData.columns[1:]:
    tData[column_name] = pd.to_numeric(tData[column_name], errors='coerce')

tData.dropna(inplace=True)  # 删除含有NaN的行
# tData.to_csv("./drop_NaN.csv")
# 读取label列唯一值
class_information = tData["label"].unique()
dataframes = {temp_class: tData[tData['label'] == temp_class] for temp_class in class_information}

# 创建列名（特征名称）空列表，然后将t检验p值小于给定值的列名填充进来
columns_index = []
for column_name in tData.columns[1:]:
    # 检验方差是否相等，并选择适当的t检验方法
    equal_var = levene(dataframes[1][column_name], dataframes[0][column_name])[1] > 0.05
    # Levene检验的p值大于0.05，各组方差相等，使用独立样本t检验
    # Levene检验的p值小于或等于0.05，各组方差不相等，使用Welch's t检验
    t_test_result = ttest_ind(dataframes[1][column_name], dataframes[0][column_name], equal_var=equal_var)[1]

    if t_test_result < 0.001:
        columns_index.append(column_name)

print(f"筛选后剩下的特征数：{len(columns_index)}个")
print(columns_index)

# 将列名label加入到columns_index列表中
if 'label' not in columns_index:
    columns_index.insert(0, 'label')

# 在分割的df中选取t检验筛选后的特征列，并将之组合为一个csv文件
df_filtered = pd.concat([dataframes[0][columns_index], dataframes[1][columns_index]])
df_filtered.to_csv('./data/rect_ttest_data.csv', header=True, index=False, encoding="utf_8_sig")
