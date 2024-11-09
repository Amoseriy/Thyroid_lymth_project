#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/8 19:02
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 创建一个简单且明确的数据矩阵
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 绘制热力图
plt.figure(figsize=(6, 6))
sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu")

# 添加标题
plt.title("Heatmap with Annotations")

# 显示图形
plt.show()