#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/11/21 16:59
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'
from PIL import Image
import os
import numpy as np

import torch
import torchvision
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNeXt50_32X4D_Weights


from torchmetrics import F1Score, Recall, Precision, Accuracy, ConfusionMatrix, Specificity
from helpers import plot
from loguru import logger
import sys

if not os.path.exists("./logs"):
    os.makedirs("./logs")


logger.remove()  # 清除默认的日志记录器，避免重复记录
# 配置日志记录器，将日志输出到文件和控制台。
# sys.stderr 用于输出到控制台，即时显示，可交互，可彩色，临时性
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO", colorize=True)
logger.add("./logs/training.log", format="{time} {level} {message}", level="INFO", colorize=False)  # 文件不能彩色输出

writer = SummaryWriter('runs/classifier_resnext50_32x4d')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100


def get_img_mean_std(root_dir, img_mode) -> tuple:
    """
    计算数据集的均值和标准差
    :param root_dir: 数据根目录
    :param img_mode: 图像模式，例如 'RGB' 或 'L'
    :return: (mean, std)
    """
    # 先创建一个空的列表用于存储每个像素的RGB值
    pixels = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                try:
                    # 遍历整个数据集，将每个像素的RGB值加入列表
                    image = Image.open(image_path).convert(img_mode)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")
                    continue
                image = np.array(image) / 255.0  # 将像素值映射到0-1范围
                # 如果是单通道图片, 则将image转换为1维数组，否则将image转换为3维数组
                pixels.append(image.flatten() if img_mode == 'L' else image.reshape(-1, 3))

    # 将像素列表转换为numpy数组
    pixels = np.concatenate(pixels, axis=0)

    # 计算每个通道的像素值的平均数和标准差
    mean = np.mean(pixels, axis=0).tolist()
    std = np.std(pixels, axis=0).tolist()

    return mean, std


def train(tra_dataloader, tes_dataloader, model, loss_fn, optimizer, epoch):
    size = len(tra_dataloader.dataset)
    model.train()

    running_loss = 0.0

    for batch, (X, y) in enumerate(tra_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        # Compute prediction error
        pred = model(X)
        # preds = pred.argmax(1)
        # logger.info(f"preds: {preds.shape}, y: {y.shape}")
        loss = loss_fn(pred, y)

        # Backpropagation
        # 计算损失函数关于所有可训练参数的梯度
        # 通过链式法则，从输出层开始逐层向前计算梯度，直到输入层。
        # loss反向传播后，梯度信息会被储存在model.parameters().grad属性中，然后被传递进优化器中
        loss.backward()
        # 根据计算出的梯度更新网络``参数``。
        optimizer.step()
        # 清空梯度，以便于下一轮计算。
        optimizer.zero_grad()

        running_loss += loss.item()

        if batch % 200 == 199:  # Every 200 mini-batches...
            logger.info(f'Batch {batch + 1}')
            # Check against the validation set
            running_vloss = 0.0

            # In evaluation mode some model specific operations can be omitted e.g. dropout layer
            model.train(False)  # Switching to evaluation mode, e.g. turning off regularisation
            for j, vdata in enumerate(tes_dataloader, 0):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
            model.train(True)  # Switching back to training mode, e.g. turning on regularisation

            avg_loss = running_loss / 1000
            avg_vloss = running_vloss / len(tes_dataloader)

            # Log the running loss averaged per batch
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch * len(tra_dataloader) + batch)

            running_loss = 0.0


        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # 初始化评价指标
    f1 = F1Score(task='binary', num_classes=2).to(DEVICE)
    recall = Recall(task='binary', num_classes=2).to(DEVICE)
    precision = Precision(task='binary', num_classes=2).to(DEVICE)
    accuracy = Accuracy(task='binary', num_classes=2).to(DEVICE)
    confusion_matrix = ConfusionMatrix(task='binary', num_classes=2).to(DEVICE)
    specificity = Specificity(task='binary', num_classes=2).to(DEVICE)

    model.eval()
    # 存储预测结果和真实标签
    all_preds = []
    all_labels = []

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            preds = pred.argmax(1)  # 将输出转换为0或1
            all_preds.append(preds)
            all_labels.append(y)

    test_loss /= num_batches
    correct /= size
    logger.info(f"Test Error:  Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} ")

    # 合并所有预测结果和真实标签
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # 计算评价指标
    f1_score = f1(all_preds, all_labels).item()
    recall_score = recall(all_preds, all_labels).item()
    precision_score = precision(all_preds, all_labels).item()
    accuracy_score = accuracy(all_preds, all_labels).item()
    confusion_matrix_score = confusion_matrix(all_preds, all_labels)
    specificity_score = specificity(all_preds, all_labels).item()


    # 输出每个epoch结束时的度量值
    logger.info(f'Epoch [{epoch+1}/{EPOCHS}] - Accuracy: {accuracy_score:.4f}, Precision: {precision_score:.4f}, Recall: {recall_score:.4f}, F1 Score: {f1_score:.4f}, Specificity: {specificity_score:.4f}')
    logger.info(f'Confusion Matrix:\n{confusion_matrix_score}')

    # 重置评价指标
    f1.reset(), recall.reset(), precision.reset(), accuracy.reset(), confusion_matrix.reset(), specificity.reset()


def main():
    ROOT_PATH = "../DATABASE_DL_JPG"
    # img_mean, img_std = get_img_mean_std(ROOT_PATH, "L")
    img_mean, img_std = [0.45615792, 0.45615792, 0.45615792], [0.27909806, 0.27909806, 0.27909806]
    print("Mean:", img_mean)  # Mean: 0.45615792
    print("Std:", img_std)  # Std: 0.27909806

    # 定义transforms
    transform = v2.Compose([
        # v2.Grayscale(num_output_channels=1),  # jpg文件是单通道的，将其转换为单通道图片
        v2.ToImage(),
        v2.Resize(256),  # 缩放到 256x256
        v2.CenterCrop(224),  # 中心裁剪到 224x224
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=(0, 180)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std),
    ])

    # 加载数据集
    dataset = ImageFolder(root=ROOT_PATH, transform=transform)
    logger.info(f"数据集分类映射：{dataset.class_to_idx}")  # 打印类别到索引的映射

    # 数据集大小
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% 训练集
    test_size = total_size - train_size

    # 按比例划分数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器

    train_loader = DataLoader(train_dataset, batch_size=28, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=28, shuffle=False, num_workers=0)

    model = torchvision.models.resnext50_32x4d(weights=None, progress=True)
    # 假设你需要将输出类别数改为 2
    num_classes = 2
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 使用tensorboard记录模型结构
    dataiter = iter(train_loader)
    images, labels = next(dataiter)  # 取出一批数据，用以构建网络结构图
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    writer.add_graph(model, images)
    writer.flush()  # 刷新缓冲区，确保数据被写入磁盘

    # 训练模型
    for t in range(EPOCHS):
        logger.info(f"{'=' * 10} Epoch {t + 1} {'=' * 10}")
        train(train_loader, test_loader,model, loss_fn, optimizer, t)
        test(test_loader, model, loss_fn, t)

        # 保存模型参数
        save_path = "./saved_models/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), f"{save_path}model_{t + 1}.pth")
        logger.info(f"Saved PyTorch Model State to {save_path}model{t + 1}.pth")

    logger.info("Done!")
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()

