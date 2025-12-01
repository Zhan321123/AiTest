import torch
import torch.nn as nn

# 定义Conv2d层（out_channels=1，其他参数按题目设置）
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)

# 构造输入：shape=(N, C, H, W) = (1, 1, 5, 5)（批量1，通道1，5×5）
x = torch.randn(1, 1, 5, 5)

# 前向传播
output = conv(x)

# 输出shape：(1, 1, 2, 2)
print("输出shape:", output.shape)  # 打印结果：torch.Size([1, 1, 2, 2])