from torch import nn
from torch.nn.functional import dropout


class AlexNet(nn.Module):
  """
  AlexNet
  inputSize: 227*227
  """
  def __init__(self):
    super(AlexNet, self).__init__()
    self.reLU = nn.ReLU()
    self.convolution1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0)
    self.maxPooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.convolution3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
    self.maxPooling4 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.convolution5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
    self.convolution6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
    self.convolution7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.maxPooling8 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(9216, 4096)
    self.linear2 = nn.Linear(4096, 4096)
    self.linear3 = nn.Linear(4096, 10)

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
    x = self.reLU(self.convolution1(x))
    x = self.maxPooling2(x)
    x = self.reLU(self.convolution3(x))
    x = self.maxPooling4(x)
    x = self.reLU(self.convolution5(x))
    x = self.reLU(self.convolution6(x))
    x = self.reLU(self.convolution7(x))
    x = self.maxPooling8(x)

    x = self.flatten(x)
    x = self.reLU(self.linear1(x))
    x = dropout(x, 0.5)
    x = self.reLU(self.linear2(x))
    x = dropout(x, 0.5)
    x = self.linear3(x)

    return x

