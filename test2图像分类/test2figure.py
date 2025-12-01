import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.python.zhan.pytorch.utils import increasePath, getResourceDir

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plotLine(trainProcess: pd.DataFrame):
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].plot(trainProcess["Epoch"], trainProcess["Train Loss"], label="Train Loss", marker="o")
  axs[0].plot(trainProcess["Epoch"], trainProcess["Val Loss"], label="Val Loss", marker="*")
  axs[0].set_xlabel("Epoch")
  axs[0].set_ylabel("Loss")
  axs[0].set_title("Loss")
  axs[0].set_ylim(0)
  axs[0].legend()
  axs[0].grid()

  axs[1].plot(trainProcess["Epoch"], trainProcess["Train Acc"], label="Train Acc", marker="o")
  axs[1].plot(trainProcess["Epoch"], trainProcess["Val Acc"], label="Val Acc", marker="*")
  axs[1].set_xlabel("Epoch")
  axs[1].set_ylabel("Acc")
  axs[1].set_title("Acc")
  axs[1].set_ylim(0, 1)
  axs[1].legend()
  axs[1].grid()

  plt.savefig(increasePath(getResourceDir() / "result/trainProcess.png"))


def plotConfusionMatrix(cm: np.ndarray, classNames: list):
  plt.figure(figsize=(12, 10))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0)
  plt.title('Confusion Matrix')
  plt.xticks(range(len(classNames)), classNames, rotation=45)
  plt.yticks(range(len(classNames)), classNames)
  plt.colorbar()
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.gca().invert_yaxis()

  plt.savefig(increasePath(getResourceDir() / "result/confusion_matrix.png"))
