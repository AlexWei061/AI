#coding = utf-8

from mxnet.gluon import nn
from mxnet import image
from d2l import mxnet as d2l
from mxnet import autograd, gluon, nd, init

net = nn.Sequential()
with net.name_scope():
    # 第一阶段
    net.add(nn.Conv2D(channels = 96, kernel_size = 11, strides = 4, activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 3, strides = 2))
    # 第二阶段
    net.add(nn.Conv2D(channels = 256, kernel_size = 5, padding = 2, activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 3, strides = 2))
    # 第三阶段
    net.add(nn.Conv2D(channels = 384, kernel_size = 3, padding = 1, activation = 'relu'))
    net.add(nn.Conv2D(channels = 384, kernel_size = 3, padding = 1, activation = 'relu'))
    net.add(nn.Conv2D(channels = 256, kernel_size = 3, padding = 1, activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 3, strides = 2))
    # 第四阶段
    net.add(nn.Flatten())
    net.add(nn.Dense(4096, activation = 'relu'))
    net.add(nn.Dropout(.5))                                                                    # 50% 的概率的丢掉 但期望不变
    # 第五阶段
    net.add(nn.Dense(4096, activation = 'relu'))
    net.add(nn.Dropout(.5))
    # 第六阶段
    net.add(nn.Dense(10))                                                                      # 真实的AlexNet是1000 但是我们这里还是用的mnist 所以就是10

print(net)

batch_size, num_epoch, lr = 128, 10, 0.01
train_data, test_data = d2l.load_data_fashion_mnist(batch_size, resize = 224)                  # resize from 28 * 28 to 224 * 224
# mnist_train = gluon.data.vision.FashionMNIST(train = True, transform = transform)
# mnist_test  = gluon.data.vision.FashionMNIST(train = True, transform = transform)
# train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle = True)
# test_data  = gluon.data.DataLoader(mnist_test,  batch_size, shuffle = False)

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

for epoch in range(num_epoch):
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
    print("Epoch %d. Loss : %f, Train acc : %f, Test acc : %f" % 
                (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))