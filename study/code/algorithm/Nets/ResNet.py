#coding = utf-8

from mxnet.gluon import nn
from d2l import mxnet as d2l
from mxnet import autograd, gluon, nd, init

class Residual(nn.Block):
    def __init__(self, channels, same_shape = True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size = 3, padding = 1, strides = strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size = 3, padding = 1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:                                                                             # 如果不一样要用一个新的层把输入也弄成残差一样的格式 因为后面要把输入和残差加起来
                self.conv3 = nn.Conv2D(channels, kernel_size = 1, strides = strides)
    
    def forward(self, x):                                                                                  # 计算残差
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)                                                                             # 输入加上残差就是输出

# blk = Residual(3)
# blk.initialize()
# x = nd.random.uniform(shape = (4, 3, 6, 6))
# y = blk(x)
# print(y.shape)
# blk2 = Residual(8, same_shape = False)
# blk2.initialize()
# y = blk2(x)
# print(y.shape)

b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size = 7, strides = 2, padding = 2),
       nn.BatchNorm(), nn.Activation('relu'),
       nn.MaxPool2D(pool_size = 3, strides = 2, padding = 1))

b2 = nn.Sequential()
b2.add(Residual(64),
       Residual(64))

b3 = nn.Sequential()
b3.add(Residual(128, same_shape = False),
       Residual(128))

b4 = nn.Sequential()
b4.add(Residual(256, same_shape = False),
       Residual(256))

b5 = nn.Sequential()
b5.add(Residual(512, same_shape = False),
       Residual(512))

b6 = nn.Sequential()
b6.add(nn.AvgPool2D(pool_size = 3), 
       nn.Dense(10))

net = nn.Sequential()
with net.name_scope():
    net.add(b1, b2, b3, b4, b5, b6)

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
        time.sleep(60)
        print("CPU has rested 1 min")