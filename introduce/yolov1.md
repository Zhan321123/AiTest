<h1 style="text-align:center;">Yolo V1</h1>

## Basic

- IOU交并比
- conf置信度
  $$conf=Pr(Object)\cdot IOU(pred,gt)$$
  - pred: 预测框
  - gt: 真实框
  - Pr(Object): 有物体=1，否则=0

## Process

- $input=Channel*Width*Height=3*448*448$
- middle
  - Conv2d
  - ReLU
  - Linear
  - MaxPool2d
- output
  $$
  \begin{align*}
  output
    &=Channel\cdot S\cdot S \\
    &=(B\times 5+C)\times 7\times 7 \\
    &=30\times 7\times 7 \\
    &=([x,y,w,h,conf]+[x,y,w,h,conf]+[class_1,class_2,...,class_{20}])\times 7\times 7 \\
  \end{align*}
  $$
- 结果认定
  - 设定置信度阈值=0.5，小于置信度的筛掉
  - 设定IOU阈值=0.5，当同一种类别的IOU大于阈值选择置信度最大的一个

## Loss

YOLOv1 损失函数 = 坐标回归损失 + 有目标置信度损失 + 无目标置信度损失 + 类别分类损失。

- 边界框坐标回归损失
  $$Loss_1=\lambda_{coord}\sum^{S^2}_{i=0}\sum^{B}_{j=0}I^{obj}_{ij}[(x_i-\hat x_i)^2+(y_i-\hat y_i)^2+(\sqrt{w_i}-\sqrt{\hat w_i})^2+(\sqrt{h_i}-\sqrt{\hat h_i})^2]$$
  - $\lambda_{coord}$: 权重=5
  - i: 遍历S*S个网格，S=7
  - j: 单个格子里的B个预测框，B=2
  - $I^{obj}_{ij}$: 指示器函数，若第 i 个网格有真实物体，且第 j 个框是该网格与真实框 IOU 最大的 “责任框”，则为 1，否则为 0
  - x,y: 真实框中心相对于所在网格的偏移量，$x,y\in[0,1]$
  - w,h: 真实框宽高相对于整个图像的比例，$w,h\in[0,1]$，加$\sqrt{\ \ }$防止更注重大目标
  - $\hat x,\hat y,\hat w,\hat h$: 预测值
- 有目标置信度损失
  $$Loss_2=\sum^{S^2}_{i=0}\sum^B_{j=0}I^{obj}_{ij}(conf_i-\hat{conf_i})^2$$
- 无目标置信度损失
  $$Loss_3=\lambda_{noobj}\sum^{S^2}_{i=0}\sum^B_{j=0}I^{noobj}_{ij}(conf_i-\hat{conf_i})^2$$
  - $\lambda_{noobj}$: 权重=0.5
  - $I^{noobj}_{ij}$: 与$I^{obj}_{ij}$相反
  - $conf_i$: 真实置信度
  - $\hat{conf_i}$: 预测置信度
- 类别分类损失
  $$Loss_4=\sum^{S^2}_{i=0}I^{obj}_{i}\sum^{C}_{c=0}(p_{ic}-\hat p_{ic})^2$$
  - C: 类别数量，C=20
  - $I^{obj}_{i}$: 若第 i 个网格有真实物体，则为 1，否则为 0
  - $p_{ic}$: 第i个网格对第c个类别的真实概率，是为1，不是为0
