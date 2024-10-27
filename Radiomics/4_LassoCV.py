#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/3/19 0:42
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# 读取数据
tData_train = pd.read_csv("./data/ttest_data.csv", encoding='utf-8-sig')

# 准备因变量（标签y）和自变量（特征X）
y = tData_train['label']
X = tData_train.drop(columns=['label'], errors='ignore', axis=1)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建Lasso回归模型
# 初始化LassoCV，设置5折交叉验证
estimator = LassoCV(cv=10, max_iter=100000, random_state=0)

# 训练模型
estimator.fit(X_train, y_train)

y_pred = estimator.predict(X_test)

# print("预测值为:\n", y_pred)
print("模型中的系数为:\n", estimator.coef_)
print("模型中的偏置为:\n", estimator.intercept_)

# 系数
features_selected = X.columns[estimator.coef_ != 0]

print("Selected features:", features_selected)
with open('lasso_features.txt', 'w') as f:
    for feature in features_selected:
        f.write(feature + '\n')

#


# ================== 绘制lasso回归路径图 ====================

# 获取cv中的alpha矩阵，根据这个矩阵计算回归路径
alphas = estimator.alphas_
coef_paths = estimator.path(X_train, y_train, alphas=alphas, max_iter=100000)

# 获取最佳alpha值，并计算它的lg，用作绘画竖线
best_alpha = np.log10(estimator.alpha_)
# 开始绘画
plt.figure(figsize=(8, 6), dpi=100)
for coef_l in coef_paths[1]:
    plt.plot(np.log10(alphas), coef_l)
plt.xlabel('log(alpha)')
plt.ylabel('Coefficients')
plt.title('Lasso Path via LassoCV')
plt.axis('tight')  # 调整图表的轴界限，使之正好包含绘图区域内的数据点，没有多余的空间。
plt.axvline(x=best_alpha, color='r', linestyle='--', label='alpha')
# plt.legend()
plt.savefig('LassoCV_coef_path.png')

# ================== 绘制lasso正则化路径图 ====================
# 提取每个alpha对应的交叉验证的测试误差的平均值和标准差
mse_mean = np.mean(estimator.mse_path_, axis=1)
mse_std = np.std(estimator.mse_path_, axis=1)

# 绘制正则化路径图
plt.figure(figsize=(8, 6), dpi=100)
plt.errorbar(np.log10(estimator.alphas_), mse_mean, yerr=mse_std)

# 标记LassoCV找到的最佳alpha（绘制一条垂直线）
plt.axvline(np.log10(estimator.alpha_), linestyle='--', color='r',
            label='alpha: CV best')

plt.title('Regularization Path')
plt.xlabel('Log(alpha)')
plt.ylabel('Mean square error')
plt.legend()
plt.savefig('LassoCV_reg_path.png')
plt.show()
