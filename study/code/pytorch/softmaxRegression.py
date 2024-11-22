import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)

net.apply(init_weights);

loss = nn.CrossEntropyLoss(reduction = 'none')

trainer = torch.optim.SGD(net.parameters(), lr = 0.1)

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.deta)
    def __getitem__(self, idx):
        return self.data[idx] 

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)                     # 如果形状是个矩阵 那么把它做成y一样的形状
    cmp = y_hat.type(y.dtype) == y                         # 预测正确的个数
    return float(cmp.type(y.dtype).sum())

def evalueate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()                                         # 评估模式
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evalueate_accuracy(net, test_iter)
        print("Epoch %d. Loss : %f, Train acc : %f, Test acc : %f" % (epoch, train_metrics[0], train_metrics[1], test_acc))

num_epochs = 10
train(net, train_iter, test_iter, loss, num_epochs, trainer)