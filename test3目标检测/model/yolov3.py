import torch
from torch import nn
from torchsummary import summary


class Convolutional(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, inputWidth):
    super(Convolutional, self).__init__()
    if stride == 1:
      padding = ((stride - 1) * inputWidth + kernel_size - stride) // 2
    elif stride == 2:
      padding = (stride * (inputWidth - 1) // 2 + kernel_size - inputWidth) // 2
    else:
      raise ValueError('stride must be 1 or 2')
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.batchNorm = nn.BatchNorm2d(out_channels)
    self.leakyRelu = nn.LeakyReLU(0.1)

  def forward(self, x):
    x = self.conv(x)
    x = self.batchNorm(x)
    x = self.leakyRelu(x)
    return x


class ConvolutionalSet(nn.Module):
  def __init__(self, in_channels, out_channels, inputWidth):
    super(ConvolutionalSet, self).__init__()
    self.convolutionalSet = nn.Sequential(
      Convolutional(in_channels, out_channels, 1, 1, inputWidth),
      Convolutional(out_channels, in_channels, 3, 1, inputWidth),
      Convolutional(in_channels, out_channels, 1, 1, inputWidth),
      Convolutional(out_channels, in_channels, 3, 1, inputWidth),
      Convolutional(in_channels, out_channels, 1, 1, inputWidth)
    )

  def forward(self, x):
    x = self.convolutionalSet(x)
    return x


class Residual(nn.Module):
  def __init__(self, in_channels, inputWidth, times):
    super(Residual, self).__init__()
    self.times = times
    self.residual = nn.Sequential(
      Convolutional(in_channels, in_channels // 2, 1, 1, inputWidth),
      Convolutional(in_channels // 2, in_channels, 3, 1, inputWidth)
    )

  def forward(self, x):
    out = None
    for i in range(self.times):
      out = self.residual(x)
    x = x + out
    return x


class DarkNet53(nn.Module):

  def __init__(self, numClasses=5):
    super(DarkNet53, self).__init__()
    self.numClasses = numClasses
    self.numAnchors = 9
    self.outputChannels = self.numAnchors * (4 + 1 + self.numClasses)

    # 主干部分
    # 416×416×3 → 416×416×32
    self.conv1 = Convolutional(3, 32, 3, 1, 416)
    # 416×416×32 → 208×208×64（下采样1）
    self.conv2 = Convolutional(32, 64, 3, 2, 416)
    # 208×208×64 → 208×208×64（1个残差块，对应1×）
    self.residual1 = Residual(64, 208, 1)
    # 208×208×64 → 104×104×128（下采样2）
    self.conv3 = Convolutional(64, 128, 3, 2, 208)
    # 104×104×128 → 104×104×128（2个残差块堆叠，对应2×）
    self.residual2 = Residual(128, 208, 2)
    # 104×104×128 → 52×52×256（下采样3）
    self.conv4 = Convolutional(128, 256, 3, 2, 104)
    # 52×52×256 → 52×52×256（8个残差块堆叠，对应8×）
    self.residual3 = Residual(256, 52, 8)
    # 52×52×256 → 26×26×512（下采样4）
    self.conv5 = Convolutional(256, 512, 3, 2, 52)
    # 26×26×512 → 26×26×512（8个残差块堆叠，对应8×）
    self.residual4 = Residual(512, 26, 8)
    # 26×26×512 → 13×13×1024（下采样5）
    self.conv6 = Convolutional(512, 1024, 3, 2, 26)
    # 13×13×1024 → 13×13×1024（4个残差块堆叠，对应4×）
    self.residual5 = Residual(1024, 13, 4)

    # 多尺度融合
    # 13×13特征融合（大目标）：13×13×1024 → 13×13×512
    self.cs1 = ConvolutionalSet(1024, 512, 13)
    # 上采样前降维：512→256
    self.conv7 = Convolutional(512, 256, 1, 1, 13)
    # 上采样：13×13→26×26（双线性插值，避免伪影）
    self.us1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    # 26×26特征融合（中目标）：拼接后 256+512=768 通道 → 26×26×256
    self.cs2 = ConvolutionalSet(768, 256, 26)
    # 上采样前降维：256→128
    self.conv8 = Convolutional(256, 128, 1, 1, 26)
    # 上采样：26×26→52×52
    self.us2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    # 52×52特征融合（小目标）：拼接后 128+256=384 通道 → 52×52×128
    self.cs3 = ConvolutionalSet(384, 128, 52)

    # 预测层
    self.predict1 = nn.Sequential(
      Convolutional(512, 1024, 3, 1, 13),
      nn.Conv2d(in_channels=1024, out_channels=self.outputChannels, kernel_size=1, stride=1, padding=0),
    )
    self.predict2 = nn.Sequential(
      Convolutional(256, 512, 3, 1, 26),
      nn.Conv2d(in_channels=512, out_channels=self.outputChannels, kernel_size=1, stride=1, padding=0),
    )
    self.predict3 = nn.Sequential(
      Convolutional(128, 256, 3, 1, 52),
      nn.Conv2d(in_channels=256, out_channels=self.outputChannels, kernel_size=1, stride=1, padding=0),
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.residual1(x)
    x = self.conv3(x)
    x = self.residual2(x)
    x = self.conv4(x)
    x = self.residual3(x)
    f3 = x
    x = self.conv5(x)
    x = self.residual4(x)
    f2 = x
    x = self.conv6(x)
    x = self.residual5(x)
    x = self.cs1(x)
    predict1 = self.predict1(x)
    x = self.conv7(x)
    x = self.us1(x)
    x = torch.cat([x, f2], dim=1)
    x = self.cs2(x)
    predict2 = self.predict2(x)
    x = self.conv8(x)
    x = self.us2(x)
    x = torch.cat([x, f3], dim=1)
    x = self.cs3(x)
    predict3 = self.predict3(x)
    return predict1, predict2, predict3


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = DarkNet53().to(device)
  summary(model, (3, 416, 416))
