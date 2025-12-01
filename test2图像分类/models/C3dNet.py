from pathlib import Path

import torch
from torch import nn
from torchsummary import summary


class C3dNet(nn.Module):
  def __init__(self):
    super(C3dNet, self).__init__()
    self.reLU = nn.ReLU()
    self.dropout = nn.Dropout(p=0.5)
    # Conv3d, (depth, height, width)
    self.convolution1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.pooling1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
    self.convolution2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.pooling2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
    self.convolution3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.convolution3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.pooling3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
    self.convolution4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.convolution4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.pooling4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
    self.convolution5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.convolution5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.pooling5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

    self.linear1 = nn.Linear(8192, 4096)
    self.linear2 = nn.Linear(4096, 4096)
    self.linear3 = nn.Linear(4096, 487)

    self.__initWeights()
    self.__loadPretrained()

  def forward(self, x):
    x = self.reLU(self.convolution1(x))
    x = self.pooling1(x)
    x = self.reLU(self.convolution2(x))
    x = self.pooling2(x)
    x = self.reLU(self.convolution3a(x))
    x = self.reLU(self.convolution3b(x))
    x = self.pooling3(x)
    x = self.reLU(self.convolution4a(x))
    x = self.reLU(self.convolution4b(x))
    x = self.pooling4(x)
    x = self.reLU(self.convolution5a(x))
    x = self.reLU(self.convolution5b(x))
    x = self.pooling5(x)

    x = x.view(-1, 8192)  # view 函数将张量x reshape成特定形状
    x = self.reLU(self.linear1(x))
    x = self.dropout(x)
    x = self.reLU(self.linear2(x))
    x = self.dropout(x)
    x = self.linear3(x)
    return x

  def __initWeights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def __loadPretrained(self, path: Path = None):
    if path is None:
      return
    modelDict = self.state_dict()
    pretrainedDict = torch.load(path)

    modelShape = [v.shape for _, v in modelDict.items()]
    pretrainedShape = [v.shape for _, v in pretrainedDict.items()]
    assert modelShape == pretrainedShape, '模型形状不一致'

    modelKeys, pretrainedKeys = list(modelDict.keys()), list(pretrainedDict.keys())
    for i in range(len(modelKeys)):
      modelDict[modelKeys[i]] = pretrainedDict[pretrainedKeys[i]]


if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = C3dNet().to(device)
  summary(model, (3, 16, 112, 112))
