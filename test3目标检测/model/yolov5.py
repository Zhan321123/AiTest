import torch
from torch import nn
from torchsummary import summary


class ConvBatchNormSilu(nn.Module):
  """
  Conv + BatchNorm + SiLU
  不下采样
  """

  def __init__(self, inChannels, outChannels, kernelSize, stride, padding):
    super(ConvBatchNormSilu, self).__init__()
    self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=stride, padding=padding)
    self.bn = nn.BatchNorm2d(outChannels)
    self.silu = nn.SiLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.silu(x)
    return x


class ResUnit(nn.Module):
  def __init__(self, inChannels, add: bool):
    super(ResUnit, self).__init__()
    self.add = add
    self.c1 = ConvBatchNormSilu(inChannels, inChannels, kernelSize=1, stride=1, padding=0)
    self.c2 = ConvBatchNormSilu(inChannels, inChannels, kernelSize=3, stride=1, padding=1)

  def forward(self, x):
    out = self.c1(x)
    out = self.c2(out)
    if self.add:
      out = out + x
    return out


class Cxxx(nn.Module):
  def __init__(self, inChannels, outChannels, resAdd: bool):
    super(Cxxx, self).__init__()
    middleChannels = outChannels // 2
    self.cbs1 = ConvBatchNormSilu(inChannels, middleChannels, kernelSize=1, stride=1, padding=0)
    self.res = ResUnit(middleChannels, resAdd)
    self.cbs2 = ConvBatchNormSilu(inChannels, middleChannels, kernelSize=1, stride=1, padding=0)
    self.cbs3 = ConvBatchNormSilu(outChannels, outChannels, kernelSize=1, stride=1, padding=0)

  def forward(self, x):
    out = self.cbs1(x)
    out = self.res(out)
    x = self.cbs2(x)
    out = torch.cat([out, x], dim=1)
    out = self.cbs3(out)
    return out


class SPPF(nn.Module):
  def __init__(self):
    super(SPPF, self).__init__()
    self.cbs1 = ConvBatchNormSilu(1024, 512, kernelSize=3, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    self.pool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    self.cbs2 = ConvBatchNormSilu(2048, 1024, kernelSize=1, stride=1, padding=0)

  def forward(self, x):
    x = self.cbs1(x)
    pool1 = self.pool1(x)
    pool2 = self.pool2(x)
    pool3 = self.pool3(x)
    out = torch.cat([x, pool1, pool2, pool3], dim=1)
    out = self.cbs2(out)
    return out


class Yolov5(nn.Module):
  """
  input: 608*608*3
  """

  def __init__(self):
    super(Yolov5, self).__init__()
    # backbone
    self.cns1 = ConvBatchNormSilu(3, 64, kernelSize=6, stride=2, padding=2)
    self.cns2 = ConvBatchNormSilu(64, 128, kernelSize=3, stride=2, padding=1)
    self.c313_1 = Cxxx(128, 128, True)
    self.cbs4 = ConvBatchNormSilu(128, 256, kernelSize=3, stride=2, padding=1)
    self.c316 = Cxxx(256, 256, True)
    self.cbs5 = ConvBatchNormSilu(256, 512, kernelSize=3, stride=2, padding=1)
    self.c319 = Cxxx(512, 512, True)
    self.cbs6 = ConvBatchNormSilu(512, 1024, kernelSize=3, stride=2, padding=1)
    self.c313_2 = Cxxx(1024, 1024, True)
    self.sppf = SPPF()
    self.cbs8 = ConvBatchNormSilu(1024, 512, kernelSize=1, stride=1, padding=0)

    # neck
    self.us3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.c323 = Cxxx(1024, 512, False)
    self.cbs2to1 = ConvBatchNormSilu(512, 256, kernelSize=1, stride=1, padding=0)
    self.us2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.c323_1 = Cxxx(512, 256, False)
    self.cbs1to2 = ConvBatchNormSilu(256, 256, kernelSize=3, stride=2, padding=1)
    self.cbsOut1 = ConvBatchNormSilu(256, 255, kernelSize=1, stride=1, padding=0)
    self.c323_2 = Cxxx(512, 512, False)
    self.cbsOut2 = ConvBatchNormSilu(512, 255, kernelSize=1, stride=1, padding=0)
    self.cbs2to3 = ConvBatchNormSilu(512, 512, kernelSize=3, stride=2, padding=1)
    self.c323_3 = Cxxx(1024, 512, False)
    self.cbsOut3 = ConvBatchNormSilu(512, 255, kernelSize=1, stride=1, padding=0)

  def forward(self, x):
    x = self.cns1(x)
    x = self.cns2(x)
    x = self.c313_1(x)
    x = self.cbs4(x)
    x = self.c316(x)
    road1 = x
    x = self.cbs5(x)
    x = self.c319(x)
    road2 = x
    x = self.cbs6(x)
    x = self.c313_2(x)
    x = self.sppf(x)
    x = self.cbs8(x)
    road3 = x

    road3to2 = self.us3(road3)
    road2 = torch.cat([road3to2, road2], dim=1)
    road2 = self.c323(road2)
    road2 = self.cbs2to1(road2)
    road2to1 = self.us2(road2)
    road1 = torch.cat([road2to1, road1], dim=1)
    road1 = self.c323_1(road1)
    out1 = self.cbsOut1(road1)
    road1to2 = self.cbs1to2(road1)
    road2 = torch.cat([road1to2, road2], dim=1)
    road2 = self.c323_2(road2)
    out2 = self.cbsOut2(road2)
    road2to3 = self.cbs2to3(road2)
    road3 = torch.cat([road2to3, road3], dim=1)
    road3 = self.c323_3(road3)
    out3 = self.cbsOut3(road3)
    return out1, out2, out3


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Yolov5().to(device)
  summary(model, (3, 608, 608))
