<h1 style="text-align:center">Activation Function</h1>

## Basic

### 模型每一次训练更新参数

$$
y=AF(wx+b) \\
\frac{\partial y}{\partial w}=AF'+x \\
\frac{\partial y}{\partial b}=AF' \\
w = w-
$$

## Common Acitivation Function

### Sigmoid

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

- 将输出值映射在(0,1)区间
- 适用于二分类任务的输出层
- 当$x\to 0$时，导数接近0，梯度消失

### Tanh

$$tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

- 解决了sigmoid的偏置问题
- 计算效率低
- x过大或过小时梯度消失
- 早期RNN在隐藏层使用

### ReLU

$$relu(x)=max(0,x)$$

- 计算极简
- x>0时导数=1，梯度永存
- x\<0时导数=0，梯度彻底消失

### Leaky ReLU

$$leakyRelu(x)=max(\alpha x,x)$$
小正数a通常取0.01

- 改进了ReLU在x\<0时的神经元死亡问题

### PReLU

$$prelu(x)=max(\alpha x,x)$$
$\alpha$不是固定值，而是可学习参数

- 比Leaky ReLU更能自适应
- 增加了模型参数量，可能导致过拟合

### ELU

$$
ELU(x)=
\begin{cases}
  x, &x>0\\
  \alpha(e^x-1), &x<0\\
\end{cases}
$$
$\alpha$通常取1

- 指数运算效率低
- 导数平滑过渡，梯度不会骤变

### SELU

$$
SELU(x)=\lambda
\begin{cases}
  x, &x>0\\
  \alpha(e^x-1), &x<0\\
\end{cases}
$$
$\lambda\approx 1.0507,\alpha\approx 1.6733$

- 实现了自归一化($\mu\approx 0,\sigma\approx 1$)，无需BatchNorm
- 仅适用于SNN网络（自归一化网络）

### GELU

$$GELU(x)=x\cdot \Phi(x)$$
$\Phi(x)$是正态分布N(0,1)的累积分布函数
GELU函数近似于$f(x)=x\cdot\sigma(1.702x)$

- 目前性能最优的激活函数之一
- 用于Transformer的隐藏层

### Swish

$$Swish(x)=x\cdot\sigma(\beta x)$$
$\beta$可为学习参数，也可为1

### Mish

$$Mish(x)=x\cdot \tanh(softplus(x))$$
$softplus(x)=\ln(1+e^x)$是ReLU的平滑版本

- 计算量略高

### Softmax

$$\sigma(x)_i=\frac{e^{z_i}}{\sum^K_{j=1}e^{z_j}}$$
K是类别数，$z_i$是第i类的logit值

- 多分类场景下使用

### Linear

$$Linnear(x)=x$$
