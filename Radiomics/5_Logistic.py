#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/3/19 23:44
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from statkit.decision import NetBenefitDisplay

# 1.读取数据
tData_train = pd.read_csv("data/ttest_data.csv", encoding='utf-8-sig')
select_features = []
with open("./data/五折/features.txt", "r") as f:
       for feature in f.readlines():
              select_features.append(str(feature).strip())
# print(select_features)
X = tData_train[select_features]
y = tData_train['label']

# 2. 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 3. 标准化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. 开始训练
estimator = LogisticRegression(max_iter=10000)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

# 5. 模型评估
print(f"原始值是：\n {list(y_test)}")
print(f"验证值是：\n {list(y_pred)}")

scores = estimator.score(X_test, y_test)
print(f"准确率是：\n {scores}")

report = classification_report(y_test, y_pred, labels=(0, 1), target_names=("良性", "恶性"))
print(report)

# 6. 绘制ROC曲线
# 6.1. 预测概率
y_scores = estimator.predict_proba(X_test)[:, 1]

# 6.2.计算ROC曲线的指标
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# 6.3.绘制ROC曲线
plt.plot(fpr, tpr, label="Logistic Regression")
# 绘制对角线
plt.plot([0, 1], [0, 1], linestyle="--")
plt.grid(linestyle="--", alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
# 6. 计算AUC值
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")

# 7.绘制校准曲线
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_scores, n_bins=10)

# 绘制校准曲线
plt.figure()
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.ylabel('Fraction of positives')
plt.xlabel('Mean predicted value')
plt.legend()
plt.show()

# 8.绘制决策曲线
y_pred_base = estimator.predict_proba(X_test)[:, 1]
# y_pred_tree = tree_model.predict_proba(X_test)[:, 1]
NetBenefitDisplay.from_predictions(y_test, y_pred_base, name='Baseline model')
# NetBenefitDisplay.from_predictions(y_test, y_pred_tree, name='Gradient boosted trees', show_references=False, ax=plt.gca())
y_ticks = np.arange(-0.05, 0.2, 0.05)
plt.yticks(y_ticks)
plt.ylim(-0.05, 0.2)
plt.show()
