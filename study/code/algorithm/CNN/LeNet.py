#coding = utf-8

from mxnet import gluon
from mxnet.gluon import nn
from d2l import mxnet as d2l
from mxnet import autograd, nd
import matplotlib.pyplot as plt

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(channels = 20, kernel_size = 5, activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 2, strides = 2))
    net.add(nn.Conv2D(channels = 50, kernel_size = 3, activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 2, strides = 2))
    net.add(nn.Flatten())
    net.add(nn.Dense(128, activation = 'relu'))
    net.add(nn.Dense(10))

ctx = d2l.try_gpu()                                                                             # 试试gpu能不能跑 如果报错则返回cpu
print("context :", ctx)
net.initialize(ctx = ctx)
print(net)

batch_size, num_epoch, lr = 256, 10, 0.5
train_data, test_data = d2l.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})

def accuracy(output, label):                                                                     # 计算拟合的准确度
    return nd.mean(output.argmax(axis = 1) == label.astype('float32')).asscalar()                # argmax是把每一列的最大概率的index返回出来 然后和label比较是否相同 最后所有求个mean

def evaluate_accuracy(data_itetator, net, context):
    acc = 0.
    for data, label in data_itetator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_itetator)

for epoch in range(num_epoch):
    train_loss, train_acc = 0., 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            out = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(out, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(out, label)
    # test_acc = 0.
    test_acc = evaluate_accuracy(test_data, net, ctx)
    print("Epoch  %d. Loss : %f, Train acc : %f, Test acc : %f" %
                (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))