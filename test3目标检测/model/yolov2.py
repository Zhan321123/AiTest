import torch
from torch import nn
from torchsummary import summary


class Convolutional(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Convolutional, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x


class Passthrough(nn.Module):
  def __init__(self):
    super(Passthrough, self).__init__()

  def forward(self, x):
    # 输入x形状：(batch, 64, 26, 26) → PyTorch默认格式：[B, C, H, W]
    batch, c, h, w = x.shape
    # reshape为 (B, 64, 13, 2, 13, 2) → 把H=26拆成13×2，W=26拆成13×2
    x = x.view(batch, c, h // 2, 2, w // 2, 2)
    # permute调整维度顺序为 (B, 64, 2, 2, 13, 13) → 把2×2的空间维度移到通道附近
    x = x.permute(0, 1, 3, 5, 2, 4)
    # reshape为 (B, 64×2×2, 13, 13) → 堆叠2×2的维度到通道，得到256通道
    x = x.contiguous().view(batch, c * 4, h // 2, w // 2)
    return x


class Yolov2(nn.Module):
  """
  input: 3*416*416
  output: 125*13*13
  """

  def __init__(self):
    super(Yolov2, self).__init__()
    self.block1 = nn.Sequential(
      Convolutional(3, 32),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block2 = nn.Sequential(
      Convolutional(32, 64),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block3 = nn.Sequential(
      Convolutional(64, 128),
      Convolutional(128, 64),
      Convolutional(64, 128),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block4 = nn.Sequential(
      Convolutional(128, 256),
      Convolutional(256, 128),
      Convolutional(128, 256),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block5 = nn.Sequential(
      Convolutional(256, 512),
      Convolutional(512, 256),
      Convolutional(256, 512),
      Convolutional(512, 256),
      Convolutional(256, 512),
    )
    self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.block6 = nn.Sequential(
      Convolutional(512, 1024),
      Convolutional(1024, 512),
      Convolutional(512, 1024),
      Convolutional(1024, 512),
      Convolutional(512, 1024),
    )
    self.block71 = nn.Sequential(
      Convolutional(1024, 1024),
      Convolutional(1024, 1024),
    )
    self.block72 = nn.Sequential(
      Convolutional(512, 64),
      Passthrough(),
    )
    self.block8 = nn.Sequential(
      Convolutional(1280, 1024),
      nn.Conv2d(1024, 125, kernel_size=1, stride=1, padding=0),
    )

  def forward(self, x):
    x = self.block1(x)  # (B,32,208,208)
    x = self.block2(x)  # (B,64,104,104)
    x = self.block3(x)  # (B,128,52,52)
    x = self.block4(x)  # (B,256,26,26)
    x = self.block5(x)  # (B,512,26,26) → 浅层高分辨率特征（给Passthrough用）

    # 深层特征分支（13×13）
    x_deep = self.maxPool(x)  # (B,512,13,13)
    x_deep = self.block6(x_deep)  # (B,1024,13,13)
    x_deep = self.block71(x_deep)  # (B,1024,13,13) → x1

    # 浅层特征分支（Passthrough融合）
    x_shallow = self.block72(x)  # (B,256,13,13) → x2_passthrough

    # 特征融合（通道维度拼接）
    x_fused = torch.cat([x_deep, x_shallow], dim=1)  # (B,1280,13,13)
    output = self.block8(x_fused)  # (B, 125,13,13)（VOC数据集）
    return output


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Yolov2().to(device)
  summary(model, (3, 416, 416))
