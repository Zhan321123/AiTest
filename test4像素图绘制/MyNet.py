import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Pixelator(nn.Module):
  def __init__(self):
    super(Pixelator, self).__init__()
    self.block1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 降维：256→128，通道3→64
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 保持128×128，增强特征
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    self.block2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128→64，通道64→128
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 保持64×64
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    self.block3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 64→32，通道128→256
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 保持32×32
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    self.block4 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 32→16，通道256→512
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 保持16×16
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True)
    )

    self.output_layer = nn.Conv2d(512, 3, kernel_size=1, stride=1, padding=0)
    # 最后用Sigmoid将输出限制在[0,1]（对应像素值归一化后范围）
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.block1(x)  # (batch_size, 64, 128, 128)
    x = self.block2(x)  # (batch_size, 128, 64, 64)
    x = self.block3(x)  # (batch_size, 256, 32, 32)
    x = self.block4(x)  # (batch_size, 512, 16, 16)
    x = self.output_layer(x)  # (batch_size, 3, 16, 16)
    x = self.sigmoid(x)  # 归一化到[0,1]
    return x


class PixelDataset(Dataset):
  def __init__(self, image256Path, image16Path, transform_input=None, transform_target=None):
    """
    加载成对的256×256输入和16×16目标图像
    :param transform_input: 输入图像的转换（针对256×256）
    :param transform_target: 目标图像的转换（针对16×16）
    """
    self.image256Path = image256Path
    self.image16Path = image16Path
    self.filenames = [f for f in os.listdir(image256Path)
                      if os.path.isfile(os.path.join(image256Path, f))]
    # 校验：确保目标文件夹中存在对应文件
    for f in self.filenames:
      if not os.path.exists(os.path.join(image16Path, f)):
        raise ValueError(f"目标文件夹缺少文件：{f}")

    self.transform_input = transform_input
    self.transform_target = transform_target

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    # 加载输入图像（256×256）
    img_name = self.filenames[idx]
    input_path = os.path.join(self.image256Path, img_name)
    input_img = Image.open(input_path).convert("RGB")  # 假设是RGB图

    # 加载目标图像（16×16）
    target_path = os.path.join(self.image16Path, img_name)
    target_img = Image.open(target_path).convert("RGB")

    # 应用转换
    if self.transform_input:
      input_img = self.transform_input(input_img)
    if self.transform_target:
      target_img = self.transform_target(target_img)

    return input_img, target_img  # 返回（输入，目标）对


def trainModelProcess(model: nn.Module, trainDataLoader: DataLoader, valDataLoader: DataLoader, numEpochs: int,
    lr: float = 0.001):  # 注意学习率通常比分类任务小
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()  # 替换为MSELoss（回归损失）
  model = model.to(device)
  bestModelWts = copy.deepcopy(model.state_dict())

  bestValLoss = float('inf')  # 跟踪最佳验证损失（越小越好）
  trainLossAll = []
  valLossAll = []
  since = time.time()

  for epoch in range(numEpochs):
    print(f" Epoch {epoch + 1}/{numEpochs} ".center(40, "="))
    trainLoss = 0.0
    valLoss = 0.0

    trainNum = 0
    valNum = 0

    model.train()  # 训练模式
    for step, (bX, bY) in enumerate(trainDataLoader):  # bX:输入(256), bY:目标(16)
      bX = bX.to(device)
      bY = bY.to(device)
      output = model(bX)  # 输出16×16图像
      loss = criterion(output, bY)  # 计算输出与目标的像素差异

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      trainLoss += loss.item() * bX.size(0)
      trainNum += bX.size(0)

    with torch.no_grad():
      model.eval()  # 验证模式
      for step, (bX, bY) in enumerate(valDataLoader):
        bX = bX.to(device)
        bY = bY.to(device)
        output = model(bX)
        loss = criterion(output, bY)

        valLoss += loss.item() * bX.size(0)
        valNum += bX.size(0)

    # 记录损失
    trainLossAvg = trainLoss / trainNum
    valLossAvg = valLoss / valNum
    trainLossAll.append(trainLossAvg)
    valLossAll.append(valLossAvg)
    print(f"{epoch}-Train Loss: {trainLossAvg:.4f}")
    print(f"{epoch}-Val Loss: {valLossAvg:.4f}")

    # 保存最佳模型（验证损失最小）
    if valLossAvg < bestValLoss:
      bestValLoss = valLossAvg
      bestModelWts = copy.deepcopy(model.state_dict())

    timeUsed = time.time() - since
    print(f"Time used: {timeUsed:.0f}s")

  # 保存模型和训练记录
  torch.save(bestModelWts, getResourceDir() / "result/myModel.pth")
  trainProcess = pd.DataFrame({
    "Epoch": range(1, numEpochs + 1),
    "Train Loss": trainLossAll,
    "Val Loss": valLossAll
  })
  trainProcess.to_csv(getResourceDir() / "result/myTrainProcess.csv", index=False)
  return trainProcess


def testModelProcess(model: nn.Module, testDataLoader: DataLoader):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  model.load_state_dict(getResourceDir() / "result/myModel.pth")
  model.eval()

  with torch.no_grad():
    for idx, (bX, bY) in enumerate(testDataLoader):
      bX = bX.to(device)
      outputs = model(bX)  # 预测的16×16图像

      # 转换为可显示的格式（从tensor→numpy，[0,1]→[0,255]）
      inputs_np = (bX.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)  # 输入256×256
      preds_np = (outputs.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)  # 预测16×16
      targets_np = (bY.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)  # 目标16×16

      # 可视化并保存
      plt.figure(figsize=(10, 4))
      plt.subplot(131)
      plt.title("Input (256×256)")
      plt.imshow(inputs_np)
      plt.subplot(132)
      plt.title("Predicted (16×16)")
      plt.imshow(preds_np)
      plt.subplot(133)
      plt.title("Target (16×16)")
      plt.imshow(targets_np)
      plt.savefig(f"{getResourceDir() / 'result/predictions'}/sample_{idx}.png")
      plt.close()

  print(f"预测结果已保存至 {getResourceDir() / 'result/predictions'}")

def preprocess_image(image_path):
  # 预处理步骤必须与训练时的input_transform完全一致！
  # （如果训练时用了Normalize，这里也要加上，参数相同）
  transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 确保输入是256×256（即使原图已是，显式处理更稳妥）
    transforms.ToTensor(),  # 转为Tensor，像素值从[0,255]→[0,1]
    # 若训练时用了归一化，这里需要添加，例如：
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  # 加载图片（确保是RGB格式，与训练数据一致）
  image = Image.open(image_path).convert("RGB")  # 若为灰度图，改为convert("L")并调整模型输入通道
  image_tensor = transform(image)  # 形状：(3, 256, 256)

  # 添加批次维度（模型要求输入是[batch_size, 3, 256, 256]）
  image_tensor = image_tensor.unsqueeze(0)  # 形状：(1, 3, 256, 256)
  return image_tensor

def predict_pixel_image(image_path, model_path, output_path):
  # 设备设置（与训练时一致，CPU/GPU）
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 加载模型结构并加载权重
  model = Pixelator()
  model.load_state_dict(torch.load(model_path, map_location=device))  # 加载model.pth
  model = model.to(device)
  model.eval()  # 切换到评估模式（关闭dropout/batchnorm的随机行为）

  # 预处理图片
  input_tensor = preprocess_image(image_path).to(device)  # 移到相同设备

  # 推理（关闭梯度计算，节省内存）
  with torch.no_grad():
    output_tensor = model(input_tensor)  # 输出形状：(1, 3, 16, 16)

  # --------------------------
  # 4. 转换输出为图片并保存
  # --------------------------
  # 移除批次维度，转为numpy数组（形状：(3, 16, 16)）
  output_np = output_tensor.squeeze(0).cpu().numpy()
  # 调整通道顺序：从[C, H, W]→[H, W, C]（适合PIL显示）
  output_np = output_np.transpose(1, 2, 0)
  # 像素值从[0,1]→[0,255]，转为uint8格式
  output_np = (output_np * 255).astype(np.uint8)

  # 保存为图片
  output_image = Image.fromarray(output_np)
  output_image.save(output_path)
  print(f"16×16像素图已保存至：{output_path}")

