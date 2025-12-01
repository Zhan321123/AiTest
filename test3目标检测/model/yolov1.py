import torch
from torch import nn
from torchsummary import summary


class Yolov1(nn.Module):
  """
  input: 448*448*3
  """

  def __init__(self):
    super(Yolov1, self).__init__()
    self.block1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=192, kernel_size=7, stride=2, padding=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block2 = nn.Sequential(
      nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.block3 = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block4 = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block5 = nn.Sequential(
      nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
    )
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(1024 * 7 * 7, 4096)
    self.dropout = nn.Dropout(0.5)
    self.linear2 = nn.Linear(4096, 7 * 7 * 30)

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.dropout(x)
    x = self.linear2(x)
    x = x.reshape(-1, 7, 7, 30)
    return x


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Yolov1().to(device)
  summary(model, (3, 448, 448))