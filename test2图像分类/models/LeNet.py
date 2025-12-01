"""
LeNet-5
  inputSize: 1*28*28

"""
import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.convolution1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
    self.sigmoid = nn.Sigmoid()
    self.averagePooling2 = nn.AvgPool2d(kernel_size=2, stride=2)
    self.convolution3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
    self.averagePooling4 = nn.AvgPool2d(kernel_size=2, stride=2)

    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(400, 120)
    self.linear2 = nn.Linear(120, 84)
    self.linear3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.sigmoid(self.convolution1(x))
    x = self.averagePooling2(x)
    x = self.sigmoid(self.convolution3(x))
    x = self.averagePooling4(x)
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    return x

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = LeNet().to(device)
  summary(model, (1, 28, 28))