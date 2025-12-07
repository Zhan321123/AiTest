<h1 style="text-align:center;">Yolov3</h1>

## Basic

- DarkNet-53
- 二分类损失函数
  - BCE，梯度友好，契合sigmoid
    $$BCE(\hat y,y)=-[y\cdot \log(\hat y)+(1-y)\log(1-\hat y)]$$
    - y: 真实标签，是该物体=1，否则=0
    - $\hat y\in[0,1]$: 预测值，已经经sigmoid转换过
  - MSE，梯度较差，惩罚力度均匀
    $$MSE(\hat y,y)=(\hat y-y)^2$$
  - RMSE
    $$RMSE=\sqrt{MSE}$$
- FPN多尺度检测与特征金字塔，划分为大中小尺度
- 平滑标签，将二分类值0/1替换为接近0/1的值(0.1/0.9)，提升泛化能力

## Precess

- $input=3*416*416$
- middle precess
  - Residual
    ```python
    out = conv2d(inChannel, inChannel//2, kernel_size=1)(input)
    out = conv2d(inChannel//2, inChannel, kernel_size=3)(out)
    return out + input
    ```
  - Up Sampling 像图像放大一样的操作
  - Convolutional
  - Convolutional Set
    ```python
    Conv2d(inChannel, outChannel, kernel_size=1)
    Conv2d(outChannel, inChannel, kernel_size=3)
    Conv2d(inChannel, outChannel, kernel_size=1)
    Conv2d(outChannel, inChannel, kernel_size=3)
    Conv2d(inChannel, outChannel, kernel_size=1)
    ```
  - Conv2d
- $output=13*13*3*85, 26*26*3*85, 52*52*3*85$
  - 3个大物体，3个中物体，3个小物体
  - 85=(x,y,w,h,conf)+80个coco数据集类别
  - 13\*13用于检测大物体，将图像划分为13\*13个区域
- 结果认定，先将所有框放在一起，然后同Yolov1

## Loss

- 整体
  $$Loss_{total}=\lambda_{coord}+Loss_{conf}+L_{cls}$$
  - $\lambda_{coord}=5$
- 边界框坐标损失
  $$Loss_{coord}=\sum^3_{j=0}\sum^{S^2}_{i=0}I^{obj}_{ij}[(\hat x_{ij}-x^*_{ij})^2+(\hat y_{ij}-y^*_{ij})^2+(\sqrt{\hat w_{ij}}-\sqrt{w^*_{ij}})^2+(\sqrt{\hat h_{ij}}-\sqrt{h^*_{ij}})^2]
  - $\sum^3_{j=0}$: 大中小框
  - S: 网格，13、26、52
  - $I^{obj}_{ij}$: 指示器函数，正样本时=1，否则=0
  - $x^*=\sigma(t_x)$
  - $w^*=p_w\cdot e^{tw}$
  - $\hat x_{ij}$: 预测值
- 物体置信度损失
  $$Loss_{conf}=\sum^3_{j=0}\sum^{S^2}_{i=0}[I^{obj}_{ij}\cdot BCE(\hat c_{ij},c^*_{ij})+\lambda_{noobj}I^{noojb}_{ij}\cdot BCE(\hat c_{ij},0)]$$
  - $\lambda_{noobj}=0.5$
  - $c^*_{ij}=IOU(pred,gt)$: 真实置信度，预测框和真实框的IOU
  - $\hat c^{ij}$: 预测值
  - $I^{noobj}_{ij}$: 负样本时=1
  - BCE(a,b): 二元交叉熵
- 类别概率损失
  $$Loss_{conf}=\sum^3_{j=0}\sum^{S^2}_{i=0}I^{obj}_{ij}\sum^{C}_{k=0}BCE(\hat p_{ijk}-p^*_{ijk})$$
  - C: 类别数量，COCO数据集=80
  - $p^*_{ijk}: 真实类别标签，是该物体=1，否则=0
  - $\hat p_{ijk}$: 预测值
