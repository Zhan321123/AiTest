import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset, Dataset

from test1基础.test3utils import *


def getDataFromFolder(folder: str,
    transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
    number: int = None) -> Dataset:
  """
  从目录提取数据
  :param folder: 分类目录子目录名为类别名，可包含其他杂文件格式:
    - folder
    - - class
    - - - element
  :param transform: 数据转换
  :param number: 样本数量，None时全部
  :return:
  """
  dataset = ImageFolder(root=folder, transform=transform)
  if number:
    dataset = Subset(dataset, range(number))
  return dataset


def splitData(dataset: Dataset, trainRatio: float = 0.8) -> (Dataset, Dataset):
  """
  数据集分割
  :param dataset: 数据集
  :param trainRatio: 训练集比例
  :return: 训练集，验证集
  """
  trainNum = int(len(dataset) * trainRatio)
  trainData, valData = random_split(dataset, [trainNum, len(dataset) - trainNum])
  return trainData, valData


def toLoader(dataset: Dataset, batchSize: int = 32, shuffle: bool = False, numWorkers: int = 0):
  """
  数据加载器
  :param dataset: 数据集
  :param batchSize: 批次大小
  :param shuffle: 是否打乱
  :param numWorkers: 工作线程数
  :return:
  """
  return DataLoader(
    dataset,
    batch_size=batchSize,
    shuffle=shuffle,
    pin_memory=True,
    # persistent_workers=True,
    num_workers=numWorkers
  )


def getMeanAndStd(dataset: Dataset)-> ([float], [float]):
  """
  计算数据集的均值和标准差
  :param dataset: 数据集
  :return: 均值和标准差
  """
  dataloader = toLoader(dataset)
  sum_channel = None
  sum_sq_channel = None
  total_pixels = 0

  for images, _ in dataloader:
    batch_size, channels, height, width = images.shape
    batch_pixels = batch_size * height * width
    total_pixels += batch_pixels

    if sum_channel is None:
      sum_channel = torch.zeros(channels)
      sum_sq_channel = torch.zeros(channels)

    sum_channel += images.sum(dim=[0, 2, 3])
    sum_sq_channel += (images ** 2).sum(dim=[0, 2, 3])

  mean = sum_channel / total_pixels
  std = torch.sqrt((sum_sq_channel / total_pixels) - (mean ** 2))
  print(f"计算得到的均值：{mean.numpy()}")
  print(f"计算得到的标准差：{std.numpy()}")
  return mean.numpy(), std.numpy()


def trainModelProcess(model: nn.Module, trainDataLoader: DataLoader, valDataLoader: DataLoader, numEpochs: int,
    lr: float = 0.01):
  """
  训练模型
  :param model:
  :param trainDataLoader:
  :param valDataLoader:
  :param numEpochs:
  :param lr: 学习率
  :return:
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器 adam，lr学习率
  criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
  model = model.to(device)
  bestModelWts = copy.deepcopy(model.state_dict())

  bestAcc = 0.0  # 最佳准确度
  trainLossAll = []  # 训练集损失
  valLossAll = []  # 验证集损失
  trainAccAll = []  # 训练集准确度
  valAccAll = []  # 验证集准确度
  since = time.time()

  for epoch in range(numEpochs):
    print(f" Epoch {epoch + 1}/{numEpochs} ".center(40, "="))
    trainLoss = 0.0  # 训练集损失
    trainAcc = 0.0  # 训练集准确度
    valLoss = 0.0  # 验证集损失
    valCorrect = 0.0  # 验证集准确度

    trainNum = 0  # 训练集样本数
    valNum = 0  # 验证集样本数

    model.train()
    for step, (bX, bY) in enumerate(trainDataLoader):
      bX = bX.to(device)
      bY = bY.to(device)
      output = model(bX)  # 前向传播
      preLab = torch.argmax(output, dim=1)
      loss = criterion(output, bY)
      optimizer.zero_grad()  # 清空梯度，否则梯度会累加
      loss.backward()  # 反向传播
      optimizer.step()  # 更新参数
      trainLoss += loss.item() * bX.size(0)
      trainAcc += torch.sum(preLab == bY.data)
      trainNum += bX.size(0)

    with torch.no_grad():
      model.eval()  # 验证模式
      for step, (bX, bY) in enumerate(valDataLoader):
        bX = bX.to(device)
        bY = bY.to(device)
        output = model(bX)

        preLab = torch.argmax(output, dim=1)
        loss = criterion(output, bY)
        valLoss += loss.item() * bX.size(0)
        valCorrect += torch.sum(preLab == bY.data)
        valNum += bX.size(0)

    trainLossAll.append(trainLoss / trainNum)
    trainAccAll.append(trainAcc.double().item() / trainNum)
    valLossAll.append(valLoss / valNum)
    valAccAll.append(valCorrect.double().item() / valNum)
    print(f"{epoch}-Train Loss: {trainLossAll[-1]:.4f} Train Acc: {trainAccAll[-1]:.4f}")
    print(f"{epoch}-Val Loss: {valLossAll[-1]:.4f} Val Acc: {valAccAll[-1]:.4f}")

    if valAccAll[-1] > bestAcc:
      bestAcc = valAccAll[-1]
      bestModelWts = copy.deepcopy(model.state_dict())

    timeUsed = time.time() - since
    print(f"Time used: {timeUsed:.0f}s")

  torch.save(bestModelWts, increasePath(getResourceDir() / "result/model.pth"))
  trainProcess = pd.DataFrame({
    "Epoch": range(1, numEpochs + 1),
    "Train Loss": trainLossAll,
    "Train Acc": trainAccAll,
    "Val Loss": valLossAll,
    "Val Acc": valAccAll
  })
  trainProcess.to_csv(increasePath(getResourceDir() / "result/trainProcess.csv"), index=False)
  return trainProcess

def testModelProcess(model: nn.Module, testDataLoader: DataLoader):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)

  testCorrect = 0
  testNum = 0
  all_preds = []  # 存储所有预测标签
  all_labels = []  # 存储所有真实标签

  with torch.no_grad():
    for step, (bX, bY) in enumerate(testDataLoader):
      bX = bX.to(device)
      bY = bY.to(device)
      model.eval()
      output = model(bX)
      preLab = torch.argmax(output, dim=1)

      all_preds.extend(preLab.cpu().numpy())
      all_labels.extend(bY.cpu().numpy())

      testCorrect += torch.sum(preLab == bY.data)
      testNum += bX.size()[0]
  testAcc = testCorrect.double().item() / testNum
  print(f"Test Acc: {testAcc:.4f}")

  cm = confusion_matrix(all_labels, all_preds)
  return cm
