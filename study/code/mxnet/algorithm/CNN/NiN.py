#coding = utf-8

from mxnet.gluon import nn
from mxnet import image
from d2l import mxnet as d2l
from mxnet import autograd, gluon, nd, init

def mlpconv(channels, kernel_size, padding, strides = 1, max_pooling = True):
    out = nn.Sequential()
    with out.name_scope():
        out.add(nn.Conv2D(channels = channels, kernel_size = kernel_size, padding = padding, strides = strides, activation = 'relu'))
        out.add(nn.Conv2D(channels = channels, kernel_size = 1, padding = 0, strides = 1, activation = 'relu'))
        out.add(nn.Conv2D(channels = channels, kernel_size = 1, padding = 0, strides = 1, activation = 'relu'))
        if max_pooling:
            out.add(nn.MaxPool2D(pool_size = 3, strides = 2))
    return out

# blk = mlpconv(64, 3, 0)
# blk.initialize()
# x = nd.random.uniform(shape = (32, 3, 16, 16))
# y = blk(x)
# print(y.shape)

net = nn.Sequential()
with net.name_scope():
    net.add(mlpconv(96, 11, 0, strides = 4))
    net.add(mlpconv(256, 5, 2))
    net.add(mlpconv(384, 3, 1))
    net.add(nn.Dropout(.5))
    net.add(mlpconv(10, 3, 1, max_pooling = False))               # 目标为10类 转成 batch_size * 10 * 5 * 5
    net.add(nn.AvgPool2D(pool_size = 5))                          #           转成 batch_size * 10 * 1 * 1
    net.add(nn.Flatten())                                         #           转成 batch_size * 10

print(net)

batch_size, num_epoch, lr = 64, 5, 0.1
train_data, test_data = d2l.load_data_fashion_mnist(batch_size, resize = 224)                  # resize from 28 * 28 to 224 * 224

ctx = d2l.try_gpu()
net.initialize(ctx = ctx, init = init.Xavier())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})

def accuracy(output, label):
    return nd.mean(output.argmax(axis = 1) == label.astype('float32')).asscalar()

def evaluate_accuracy(data_itetator, net, context):
    acc = 0.
    for data, label in data_itetator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_itetator)

import time

for epoch in range(num_epoch):
    start_time = time.time()
    train_loss, train_acc = 0., 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        data = data.as_in_context(ctx)
        with autograd.record():
            out = net(data)
            loss = softmax_cross_entropy(out, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(out, label)
    test_acc = evaluate_accuracy(test_data, net, ctx)
    end_time = time.time()
    print("Epoch %d. Loss : %f, Train acc : %f, Test acc : %f, Used time : %f min" % 
                (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc, (end_time - start_time) / 60))
    if(epoch != num_epoch - 1):
        time.sleep(180)
        print("CPU has rested 3 min")