<h1 style="text-align:center">Bag of Freebies and Specials</h1>

## 数据增强

- MixUp
  两张图片虚化并重叠在一起，设定$(x_1,y_1,w_1,h_1,conf_1)+(x_2=0.5,y_2=0.5,...)+(c_1,c_2,0,0,...)$
- Cutout
  去掉一个物体的某一块内容成黑块/白块，设定(x,y,w-w_cut,h-h_cut)+(c=0.6,0,0,...)
- CutMix
  Cutout的增强版，两个物体各取一部分拼接在一起，没有黑块/白块
- Mosaic
  CutMix里2个物体拼接，Mosaic里4个物体

### 余弦退火算法

动态调整学习率，同时避免模型陷入最优解
$$lr(t)=lr_{min}+\frac{1}{2}(lr_{max}-lr_{min})(1+cos(\frac{T_{cur}}{T_{max}}\pi))$$

- lr(t): 第t次模型训练的学习率
- $lr_{max}$: 初始学习率
- $lr_{min}=lr(T_{max})$: 最小学习率
- $T_{cur}: 当前训练的epoch数
- $T_{max}: 总的退火周期/总的训练次数
