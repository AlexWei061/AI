# coding = utf-8

from mxnet import autograd, np, npx

npx.set_np()

x = np.arange(4)
print("x : ", x)
x.attach_grad()                                   # 通过调用attach_grad来为一个张量的梯度分配内存
print("x.grad : ", x.grad, '\n')                  # 在计算关于x的梯度后，将能够通过'grad'属性访问它，它的值被初始化为0

def f1(x):                                        # f1(x) = 2 * x^2
    return 2 * np.dot(x, x)

def f2(x):
    return x.sum()

def f3(x):
    return x * x

def f4(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

with autograd.record():                           # 把代码放到autograd.record内，以建立计算图
    y = f1(x)

print("y : ", y, '\n')

y.backward()                                      # 反向传播函数来自动计算y关于x每个分量的梯度
print("x.grad : ", x.grad)
print("4x : ", 4 * x)
print("is equal : ", x.grad == 4 * x, '\n')       # 看看算对没有


# 第二个函数
with autograd.record():
    y = f2(x)
y.backward()
print(x.grad, '\n')

# 第三个函数
with autograd.record():
    y = f3(x)                                     # 这个f3(x)输出的是一个向量
y.backward()
print(x.grad, '\n')                               # 这里等价于 y = sum(x * x)

# 第四个函数
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f4(a)
d.backward()
print(a.grad)
print(a.grad == d / a)