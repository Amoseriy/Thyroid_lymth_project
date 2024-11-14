#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/3/18 22:17
"""
再对benign.csv和malign.csv文件进行处理,去掉字符串特征，插入label标签。
malign标签为1，benign标签为0
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

original_csv = pd.read_csv("./data/RECT_LABEL/rect_combined_total_features.csv", low_memory=False)
# 读取提取的CSV特征数据
# 根据Label列的最后一个字符进行筛选和保存
benign_data = original_csv[original_csv['Label'].str.endswith('_B')]
mal_data = original_csv[original_csv['Label'].str.endswith('_M')]


# 在第一列插入一行列名为label的列，良性的值为0，恶性为1
benign_data.insert(0, 'label', 0)  # 插入标签
mal_data.insert(0, 'label', 1)  # 插入标签

# 读取每一列第一行的数据，如果是str，则将列名记录下来，然后删除整列

benign_cols = [x for i, x in enumerate(benign_data.columns) if isinstance(benign_data.iloc[0, i], str)]  # 了解列表生成式
benign_data = benign_data.drop(benign_cols, axis=1)

mal_cols = [x for i, x in enumerate(mal_data.columns) if isinstance(mal_data.iloc[0, i], str)]
mal_data = mal_data.drop(mal_cols, axis=1)

# 再合并成一个新的csv文件。
total_data = pd.concat([benign_data, mal_data])
total_data.to_csv('./data/RECT_LABEL/rect_FilteredTotalData.csv', index=False)

# 简单查看数据的分布

# 创建空白画布fig和坐标轴ax
fig, ax = plt.subplots()
# 设置为sns默认绘画风格
sns.set()
# 使用sns.countplot()绘制计数柱状图，数据是total_data，列名是label
# countplot会计算每个唯一值出现的次数，并显示为一个柱状图。
# 这里将图绘制在前面创建的ax坐标轴上。
ax = sns.countplot(x='label', data=total_data)
plt.show()
# 计算label列的值的个数
print(total_data['label'].value_counts())
