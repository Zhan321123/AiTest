import torch
from torch import nn
from torchsummary import summary


def makeDivisible(x, divisor=8):
  return int(((x + divisor - 1) // divisor) * divisor)


class ConvBatchNormSilu(nn.Module):
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
  def __init__(self, inChannels, outChannels, resAdd: bool, numRes: int):
    super(Cxxx, self).__init__()
    middleChannels = outChannels // 2
    self.cbs1 = ConvBatchNormSilu(inChannels, middleChannels, kernelSize=1, stride=1, padding=0)
    self.res = nn.Sequential(*[ResUnit(middleChannels, resAdd) for _ in range(numRes)])
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
  def __init__(self, inChannels, midChannels, outChannels):
    super(SPPF, self).__init__()
    self.cbs1 = ConvBatchNormSilu(inChannels, midChannels, kernelSize=3, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    self.pool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    self.cbs2 = ConvBatchNormSilu(midChannels * 4, outChannels, kernelSize=1, stride=1, padding=0)

  def forward(self, x):
    x = self.cbs1(x)
    pool1 = self.pool1(x)
    pool2 = self.pool2(x)
    pool3 = self.pool3(x)
    out = torch.cat([x, pool1, pool2, pool3], dim=1)
    out = self.cbs2(out)
    return out


class Yolov5(nn.Module):
  def __init__(self, depthMultiple=1.0, widthMultiple=1.0):
    super(Yolov5, self).__init__()
    baseChannel = {
      'cns1_out': 64, 'cns2_out': 128, 'c313_1_out': 128,
      'cbs4_out': 256, 'c316_out': 256, 'cbs5_out': 512,
      'c319_out': 512, 'cbs6_out': 1024, 'c313_2_out': 1024,
      'sppf_mid': 512, 'cbs8_out': 512, 'c323_out': 512,
      'cbs2to1_out': 256, 'c323_1_out': 256, 'cbs1to2_out': 256,
      'c323_2_out': 512, 'cbs2to3_out': 512, 'c323_3_out': 512
    }
    scaledChannel = {k: makeDivisible(v * widthMultiple, 8) for k, v in baseChannel.items()}

    def scaleDepth(num):
      return max(round(num * depthMultiple), 1)

    # backbone
    self.cns1 = ConvBatchNormSilu(3, scaledChannel['cns1_out'], kernelSize=6, stride=2, padding=2)
    self.cns2 = ConvBatchNormSilu(scaledChannel['cns1_out'], scaledChannel['cns2_out'], kernelSize=3, stride=2,
      padding=1)
    self.c313_1 = Cxxx(scaledChannel['cns2_out'], scaledChannel['c313_1_out'], True, scaleDepth(3))
    self.cbs4 = ConvBatchNormSilu(scaledChannel['c313_1_out'], scaledChannel['cbs4_out'], kernelSize=3, stride=2,
      padding=1)
    self.c316 = Cxxx(scaledChannel['cbs4_out'], scaledChannel['c316_out'], True, scaleDepth(6))
    self.cbs5 = ConvBatchNormSilu(scaledChannel['c316_out'], scaledChannel['cbs5_out'], kernelSize=3, stride=2,
      padding=1)
    self.c319 = Cxxx(scaledChannel['cbs5_out'], scaledChannel['c319_out'], True, scaleDepth(9))
    self.cbs6 = ConvBatchNormSilu(scaledChannel['c319_out'], scaledChannel['cbs6_out'], kernelSize=3, stride=2,
      padding=1)
    self.c313_2 = Cxxx(scaledChannel['cbs6_out'], scaledChannel['c313_2_out'], True, scaleDepth(3))
    # SPPF传入缩放后的通道数（不再硬编码）
    self.sppf = SPPF(scaledChannel['c313_2_out'], scaledChannel['sppf_mid'], scaledChannel['c313_2_out'])
    self.cbs8 = ConvBatchNormSilu(scaledChannel['c313_2_out'], scaledChannel['cbs8_out'], kernelSize=1, stride=1,
      padding=0)

    # neck
    self.us3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    # 拼接后通道数 = 上采样通道 + 骨干网络对应层通道
    self.c323 = Cxxx(scaledChannel['cbs8_out'] + scaledChannel['c319_out'], scaledChannel['c323_out'], False,
      scaleDepth(3))
    self.cbs2to1 = ConvBatchNormSilu(scaledChannel['c323_out'], scaledChannel['cbs2to1_out'], kernelSize=1, stride=1,
      padding=0)
    self.us2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.c323_1 = Cxxx(scaledChannel['cbs2to1_out'] + scaledChannel['c316_out'], scaledChannel['c323_1_out'], False,
      scaleDepth(3))
    self.cbs1to2 = ConvBatchNormSilu(scaledChannel['c323_1_out'], scaledChannel['cbs1to2_out'], kernelSize=3, stride=2,
      padding=1)
    self.cbsOut1 = ConvBatchNormSilu(scaledChannel['c323_1_out'], 255, kernelSize=1, stride=1,
      padding=0)  # 输出通道255固定（80类+坐标+置信）
    self.c323_2 = Cxxx(scaledChannel['cbs1to2_out'] + scaledChannel['cbs2to1_out'], scaledChannel['c323_2_out'], False,
      scaleDepth(3))
    self.cbsOut2 = ConvBatchNormSilu(scaledChannel['c323_2_out'], 255, kernelSize=1, stride=1, padding=0)
    self.cbs2to3 = ConvBatchNormSilu(scaledChannel['c323_2_out'], scaledChannel['cbs2to3_out'], kernelSize=3, stride=2,
      padding=1)
    self.c323_3 = Cxxx(scaledChannel['cbs2to3_out'] + scaledChannel['cbs8_out'], scaledChannel['c323_3_out'], False,
      scaleDepth(3))
    self.cbsOut3 = ConvBatchNormSilu(scaledChannel['c323_3_out'], 255, kernelSize=1, stride=1, padding=0)

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
  yolov5Params = {
    'medium': (0.67, 0.75),
    'large': (1.00, 1.00),
    'xlarge': (1.33, 1.25),
    'small': (0.33, 0.50),
    'nano': (0.33, 0.25)
  }

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Yolov5(*yolov5Params['nano']).to(device)
  summary(model, (3, 608, 608))

  test_input = torch.randn(1, 3, 608, 608).to(device)
  out1, out2, out3 = model(test_input)
  print(f"输出:")
  print(f"out1 (76): {out1.shape}")
  print(f"out2 (38): {out2.shape}")
  print(f"out3 (19): {out3.shape}")
