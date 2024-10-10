# coding = utf-8

from mxnet.gluon import nn
from d2l import mxnet as d2l
from mxnet import autograd, gluon, nd, init

def conv_block(channels):
    out = nn.Sequential()
    out.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels, kernel_size = 3, padding = 1))
    return out

class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim = 1)                                     # 把残差网络中的加法改成了concat
        return x
    
# blk = DenseBlock(2, 10)
# blk.initialize()
# x = nd.random.uniform(shape = (4, 3, 8, 8))
# print(blk(x).shape)

def trasition_block(channels):                                                  # 因为一直concat会导致channels变得巨大无比 所以我们来给它变一下形
    out = nn.Sequential()
    out.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels, kernel_size = 1),                               # channels变成目标channels
            nn.AvgPool2D(pool_size = 2, strides = 2))
    return out

growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10

net = nn.Sequential()
with net.name_scope():
    # first block
    net.add(nn.Conv2D(64, kernel_size = 7, strides = 2, padding = 3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size = 3, strides = 2, padding = 1))
    
    #dense block
    channels = 64
    for i, layers in enumerate(block_layers):
        net.add(DenseBlock(layers, growth_rate))
        channels += layers * growth_rate
        if i != len(block_layers) - 1:
            net.add(trasition_block(channels // 2))
    
    # last block
    net.add(nn.BatchNorm(),
             nn.Activation('relu'),
             nn.AvgPool2D(pool_size = 1),
             nn.Flatten(),
             nn.Dense(10))

print(net)

batch_size, num_epoch, lr = 128, 5, 0.1
train_data, test_data = d2l.load_data_fashion_mnist(batch_size, resize = 96)                  # resize from 28 * 28 to 96 * 96

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
        time.sleep(300)
        print("CPU has rested 5 min")