import torch

t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
t2 = t1 + 2

print(t1)
print(t2)

print("----------")
print(t1.shape)
print(t1.dtype)
print(t1.device)
print(t1.requires_grad)
print(t1.grad_fn)
print(t1.is_leaf)
print(t1.is_cuda)
print(t1.is_sparse)

print('----------')
print(t1.size())
print(t1.size(0))
print(t1.numel())
print(t1.is_contiguous())
print(t1.is_pinned())

print('-------')
print(torch.numel(t1))

print('---------')
print(t1 @ t1)
print(torch.matmul(t1, t1))
print(torch.mm(t1, t1))
print(t1.matmul(t1))
print(t1.mm(t1))