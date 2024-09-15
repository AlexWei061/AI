# coding = utf-8

from mxnet.gluon import nn
from mxnet import image
from d2l import mxnet as d2l
from mxnet import autograd, gluon, nd, init

class Inception(nn.Block):
    def __init__(self, n1, n2, n3, n4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        with self.name_scope():
            # path 1
            self.p1Conv1 = nn.Conv2D(n1,    kernel_size = 1, activation = 'relu')
            # path 2
            self.p2Conv1 = nn.Conv2D(n2[0], kernel_size = 1, activation = 'relu')
            self.p2Conv3 = nn.Conv2D(n2[1], kernel_size = 3, padding = 1, activation = 'relu')
            # path 3
            self.p3Conv1 = nn.Conv2D(n3[0], kernel_size = 1, activation = 'relu')
            self.p3Conv5 = nn.Conv2D(n3[1], kernel_size = 5, padding = 2, activation = 'relu')
            # path 4
            self.p4Pool3 = nn.MaxPool2D(pool_size = 3, padding = 1, strides = 1)
            self.p4Conv1 = nn.Conv2D(n4,    kernel_size = 1, activation = 'relu')
    
    def forward(self, x):
        p1 = self.p1Conv1(x)
        p2 = self.p2Conv3(self.p2Conv1(x))
        p3 = self.p3Conv5(self.p3Conv1(x))
        p4 = self.p4Conv1(self.p4Pool3(x))
        return nd.concat(p1, p2, p3, p4, dim = 1)                          # 是(batch_size, channels, height, width) 且是按照channels来concat 所以dim=1
    
# incp = Inception(64, (96, 128), (16, 32), 32)
# incp.initialize()
# x = nd.random.uniform(shape = (32, 3, 64, 64))
# y = incp(x)
# print(y.shape)

# block 1
b1 = nn.Sequential()
with b1.name_scope():
    b1.add(nn.Conv2D(64, kernel_size = 7, strides = 2, padding = 3, activation = 'relu'))
    b1.add(nn.MaxPool2D(pool_size = 3, strides = 2, padding = 1))

# block 2
b2 = nn.Sequential()
with b2.name_scope():
    b2.add(nn.Conv2D(64, kernel_size = 1, activation = 'relu'))
    b2.add(nn.Conv2D(192, kernel_size = 3, padding = 1, activation = 'relu'))
    b2.add(nn.MaxPool2D(pool_size = 3, strides = 2, padding = 1))

# block 3
b3 = nn.Sequential()
with b3.name_scope():
    b3.add(Inception(64, (96, 128), (16, 32), 32))
    b3.add(Inception(128, (128, 192), (32, 96), 64))
    b3.add(nn.MaxPool2D(pool_size = 3, strides = 2, padding = 1))

# block 4
b4 = nn.Sequential()
with b3.name_scope():
    b4.add(Inception(192, (96, 208), (16, 48), 64))
    b4.add(Inception(160, (112, 224), (24, 64), 64))
    b4.add(Inception(128, (128, 256), (24, 64), 64))
    b4.add(Inception(112, (144, 288), (32, 64), 64))
    b4.add(Inception(256, (160, 320), (32, 128), 128))
    b4.add(nn.MaxPool2D(pool_size = 3, strides = 2, padding = 1))

# block 5
b5 = nn.Sequential()
with b5.name_scope():
    b5.add(Inception(256, (160, 320), (32, 128), 128))
    b5.add(Inception(384, (192, 384), (48, 128), 128))
    b5.add(nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))

print(net)

# x = nd.random.uniform(shape = (4, 3, 96, 96))
# y = net(x)
# print(y.shape)

batch_size, num_epoch, lr = 128, 5, 0.01
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
        time.sleep(60)
        print("CPU has rested 1 min")