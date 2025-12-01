"""
VggNet
inputSize: 224*224
"""
from torch import nn


class VggNet(nn.Module):
  def __init__(self):
    super(VggNet, self).__init__()
    self.block1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block3 = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block4 = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block5 = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block6 = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Linear(4096, 10)
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
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    return x

