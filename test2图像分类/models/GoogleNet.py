import torch
from torch import nn

class InceptionNet(nn.Module):
  """
  InceptionNet
  """
  def __init__(self, in_channels, c1, c2, c3, c4):
    super(InceptionNet, self).__init__()
    self.reLU = nn.ReLU()
    self.p11 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
    self.p21 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
    self.p22 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
    self.p31 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
    self.p32 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
    self.p41 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    self.p42 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

  def forward(self, x):
    p1 = self.reLU(self.p11(x))
    p2 = self.reLU(self.p22(self.reLU(self.p21(x))))
    p3 = self.reLU(self.p32(self.reLU(self.p31(x))))
    p4 = self.reLU(self.p42(self.p41(x)))
    return torch.cat([p1, p2, p3, p4], dim=1)


class GoogleNet(nn.Module):
  """
  GoogleNet
  inputSize: 224*224
  """
  def __init__(self):
    super(GoogleNet, self).__init__()
    self.b1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.b2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.b3 = nn.Sequential(
      InceptionNet(192, 64, (96, 128), (16, 32), 32),
      InceptionNet(256, 128, (128, 192), (32, 96), 64),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.b4 = nn.Sequential(
      InceptionNet(480, 192, (96, 208), (16, 48), 64),
      InceptionNet(512, 160, (112, 224), (24, 64), 64),
      InceptionNet(512, 128, (128, 256), (24, 64), 64),
      InceptionNet(512, 112, (128, 288), (32, 64), 64),
      InceptionNet(528, 256, (160, 320), (32, 128), 128),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.b5 = nn.Sequential(
      InceptionNet(832, 256, (160, 320), (32, 128), 128),
      InceptionNet(832, 384, (192, 384), (48, 128), 128),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(1024, 2)
    )

    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)

  def forward(self, x):
    x = self.b1(x)
    x = self.b2(x)
    x = self.b3(x)
    x = self.b4(x)
    x = self.b5(x)
    return x
