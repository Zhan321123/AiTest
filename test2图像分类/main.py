import torch
from torch import nn
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST

from init import Root
from .test2figure import plotLine, plotConfusionMatrix
from modelTrain import splitData, toLoader, trainModelProcess, testModelProcess, getDataFromFolder


def summaryModel(model: nn.Module, inputSize: tuple):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  summary(model, inputSize)


def train(model: nn.Module):
  transformer = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化到[-1, 1]
  ])
  # trainData = FashionMNIST(root=getResourceDir() / "data/fashion",
  trainData = MNIST(root=Root / "data/number",
    train=True,
    download=False,
    transform=transformer)
  # trainData = getDataFromFolder(getResourceDir() / "data/PetImages", number=600,transform=transformer)
  trainData, valData = splitData(trainData)
  trainLoader = toLoader(trainData)
  valLoader = toLoader(valData)
  trainProcess = trainModelProcess(model, trainLoader, valLoader, 6, 0.001)
  plotLine(trainProcess)


def test(model: nn.Module):
  testData = FashionMNIST(root=Root / "data/fashion",
    # testData = MNIST(root=getResourceDir() / "data/number",
    train=False,
    download=False,
    transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]))
  testLoader = toLoader(testData)
  model.load_state_dict(torch.load(Root / "result/model(2).pth"))
  cm = testModelProcess(model, testLoader)
  plotConfusionMatrix(cm, MNIST.classes)


if __name__ == '__main__':
  model = NoConvNet()
  # summaryModel(model)
  train(model)
  # test(model)
