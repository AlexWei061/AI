# coding=utf-8
import torch

x = torch.arange(12)
print("x :")
print(x)
print(x.shape)
print(x.numel())
print("")

X = x.reshape(3, 4)
print("X :")
print(X)
print(X.shape)
print("")

print(torch.zeros((2, 3, 4)))
print("")
print(torch.ones((2, 3, 4)))
print("")

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** 2)
print("")

x = torch.arange(12, dtype = torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((x, y), dim = 0))
print(torch.cat((x, y), dim = 1))
print("")
print(x == y)
print(x.sum())
print("")

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print("")
print(a + b)                                   # 广播
print("")

before = id(y)                                 # 新建内存
y = y + x
print(id(y) == before)
print("")

z  = torch.zeros_like(y)
print("id(z) :", id(x))
z[:] = x + y
print("id(z) :", id(x))
print("")

before = id(x)                                 # 原地操作
x += y
print(id(x) == before)

A = x.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))
print("")

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))