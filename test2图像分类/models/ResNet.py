import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
  """
  残差块
  """
  def __init__(self, inChannels, outChannels, stride=1, use1Conv=False):
    super(Residual, self).__init__()
    self.reLU = nn.ReLU()
    self.convolution1 = nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, stride=stride,padding=1)
    self.convolution2 = nn.Conv2d(in_channels=outChannels, out_channels=outChannels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(outChannels)
    self.bn2 = nn.BatchNorm2d(outChannels)
    if use1Conv:
      self.convolution3 = nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=1, stride=stride)
    else:
      self.convolution3 = None

  def forward(self, x):
    out = self.reLU(self.bn1(self.convolution1(x)))
    out = self.bn2(self.convolution2(out))
    if self.convolution3:
      x = self.convolution3(x)
    out += x
    out = self.reLU(out)
    return out


class ResNet(nn.Module):
  """
  ResNet
  inputSize: 224*224
  """
  def __init__(self):
    super(ResNet, self).__init__()
    self.block1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.block2 = nn.Sequential(
      Residual(64, 64),
      Residual(64, 64)
    )
    self.block3 = nn.Sequential(
      Residual(64, 128, stride=2, use1Conv=True),
      Residual(128, 128)
    )
    self.block4 = nn.Sequential(
      Residual(128, 256, stride=2, use1Conv=True),
      Residual(256, 256)
    )
    self.block5 = nn.Sequential(
      Residual(256, 512, stride=2, use1Conv=True),
      Residual(512, 512)
    )
    self.block6 = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(512, 10)
    )

  def forward(self, x):
    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    out = self.block4(out)
    out = self.block5(out)
    out = self.block6(out)
    return out

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = ResNet().to(device)
  summary(model, (1, 224, 224))