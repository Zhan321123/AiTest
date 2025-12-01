import torch
import torch.version

print(torch.cuda.is_available())  # torch的cuda是否可用

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # 打印设备
print(torch.cuda.get_device_name(0))  # 打印GPU0的名字
print(torch.rand(3, 3).cuda())  # 创建一个3*3的随机张量，并将其分配到GPU0

print(torch.version.cuda)  # cuda的版本
print(torch.backends.cudnn.is_available())  # cudnn是否可用
print(torch.backends.cudnn.version())  # cudnn的版本
