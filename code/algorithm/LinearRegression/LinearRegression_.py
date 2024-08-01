# coding = utf-8

import random
from mxnet import np, npx, autograd
from d2l import mxnet as d2l


n = 1000
true_w = np.array([3.7, -5.2])
true_b = 4
bat_size = 10

def gen_date(n):
    x = np.random.normal(0, 1, (n, 2))
    y = np.dot(x, true_w) + true_b
    y += np.random.normal(0, 0.01, y.shape)
    return x, y

def date_iter(bat_size, xs, ys):                              # 随机选batch size个出来
    num = len(xs)
    # print(num)
    ind = list(range(num))
    random.shuffle(ind)
    # print(ind)
    feats = np.zeros((bat_size, 2))
    labs = np.zeros(bat_size)
    for i in range(bat_size):
        feats[i] = xs[ind[i]]
        labs[i] = ys[ind[i]]
    return feats, labs

def h(w, b, x):
    return np.dot(x, w) + b

def cost(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def SGD(paras, lr, bat_size):
    for para in paras:
        para[:] = para - lr * para.grad;


xs, ys = gen_date(n)
# f, l = date_iter(bat_size, xs, ys)
# print(xs)
# print(ys)
# print(f)
# print(l)
times = 20
lr = 0.05
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

for k in range(times):
    f, l = date_iter(bat_size, xs, ys)
    # print(f)
    # print(l)
    with autograd.record():
        J = cost(h(w, b, f), l)
    J.backward()
    SGD([w, b], lr, bat_size)
    c = cost(h(w, b, xs), ys)
    print("第 ", k + 1, " 次计算 : cost = ", c.mean())

print("true w : ", true_w)
print("true b : ", true_b)
print("w : ", w.reshape(1, 2))
print("b : ", b)