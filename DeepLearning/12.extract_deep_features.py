#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2025/1/10 20:41
import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练的 VGG16 模型
model = models.vgg16(pretrained=False)  # 不使用预训练权重

# 修改模型，移除最后的全连接层（分类层）
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # 保留除最后一层外的所有层

# 加载你训练好的权重
model.load_state_dict(torch.load('../DATABASE_PTH/vgg_best_model.pth'))
model.eval()  # 设置为评估模式