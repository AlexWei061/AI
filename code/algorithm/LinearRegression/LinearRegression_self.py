# coding = utf-8

import random
from mxnet import np, npx, autograd
from d2l import mxnet as d2l

npx.set_np()

lr = 0.03
training_size = 1000
true_w = np.array([3.4, -5.2])
true_b = 4
batch_size = 10
times = 10

def gen_date(training_size):
    xs = np.random.normal(0, 1, (training_size, 2))
    ys = np.dot(xs, true_w.reshape(2, 1)) + true_b
    ys += np.random.normal(0, 0.01, ys.shape)
    return xs, ys

def date_iter(batch_size, xs, ys):
    size = len(xs)
    ind = list(range(size))
    random.shuffle(ind)
    features = np.zeros((batch_size, 2))
    lables = np.zeros(batch_size)
    for i in range(batch_size):
        features[i] = xs[ind[i]]
        lables[i] = ys[ind[i]]
    return features, lables

def h(w, b, x):
    return np.dot(x, w) + b

def cost(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

xs, ys = gen_date(training_size)
# print(xs)
# print(ys)

w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

for i in range(times):
    x, y = date_iter(batch_size, xs, ys)
    with autograd.record():
        J = cost(h(w, b, x), y)
    J.backward()
    w[:] -= lr * w.grad
    b[:] -= lr * b.grad
    trainedJ = cost(h(w, b, xs), ys).mean()
    print("第 ", i + 1, " 此训练: cost = ", trainedJ)

print("true w : ", true_w)
print("true b : ", true_b)
print("w : ", w.reshape(1, 2))
print("b : ", b)