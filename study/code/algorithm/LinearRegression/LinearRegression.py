import random
from mxnet import np, npx, autograd
from d2l import mxnet as d2l

npx.set_np()

true_w = np.array([2, -3.4])                                     # 真实的 w 和 b
true_b = 4.2
bat_size = 10

def gen_date(w, b, n):
    x = np.random.normal(0, 1, (n, len(w)))
    # print("w :\n", w)
    # print("x :\n", x)
    y = np.dot(x, w) + true_b
    # print("y :\n", y)
    y += np.random.normal(0, 0.01, y.shape)
    # print("y :\n", y)
    return x, y

def date_iter(bat_size, feats, labs):
    n = len(feats)
    ind = list(range(n))                                          # 生成一个这样的数组：[0, 1, 2, ..., n-1]
    random.shuffle(ind)                                           # 打乱
    for i in range(0, n, bat_size):
        bat_ind = np.array(ind[i : min(i + bat_size, n)])
        # print("bat_ind :\n", bat_ind)
        yield feats[bat_ind], labs[bat_ind]

feats, labs = gen_date(true_w, true_b, 1000)

# for x, y in date_iter(bat_size, feats, labs):
#     print("x :\n", x)
#     print("y :\n", y)
#     break

# 初始化
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

def h(w, b, x):
    return np.dot(x, w) + b

def cost(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def SGD(paras, lr, bat_size):
    for para in paras:
        para[:] = para - lr * para.grad / bat_size

lr = 0.03
num_epochs = 10

for epoch in range(num_epochs):
    for x, y in date_iter(bat_size, feats, labs):
        with autograd.record():
            J = cost(h(w, b, x), y)
        J.backward()
        SGD([w, b], lr, bat_size)
    trainedJ = cost(h(w, b, feats), labs)
    print(f'epoch {epoch + 1}, loss {float(trainedJ.mean()):f}')

print("true w : ", true_w)
print("true b : ", true_b)
print("w : ", w.reshape(1, 2))
print("b : ", b)