# coding = utf-8

from mxnet import gluon
from mxnet.gluon import nn
from d2l import mxnet as d2l
from mxnet import autograd, nd
import matplotlib.pyplot as plt

def transform(data, label):                                                              # 把图片转成矩阵的函数
    return data.astype('float32') / 255, label.astype('float32')
mnist_train = gluon.data.vision.FashionMNIST(train = True, transform = transform)
mnist_test  = gluon.data.vision.FashionMNIST(train = True, transform = transform)

def get_text_labels(label):
    text_labels = [ 't-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot' ]
    return [text_labels[int(i)] for i in label]

# 试试show_images的效果
# data, label = mnist_train[0 : 9]
# print(get_text_labels(label))
# show_images(data)

# Load 数据
batch_size, num_epoch, lr = 256, 10, 0.1
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle = True)
test_data  = gluon.data.DataLoader(mnist_test,  batch_size, shuffle = False)

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(10))
net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})

def accuracy(output, label):                                                     # 计算拟合的准确度
    return nd.mean(output.argmax(axis = 1) == label).asscalar()                  # argmax是把每一列的最大概率的index返回出来 然后和label比较是否相同 最后所有求个mean

def evaluate_accuracy(data_itetator, net):
    acc = 0.
    for data, label in data_itetator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_itetator)

for epoch in range(num_epoch):
    train_loss, train_acc = 0., 0.
    for data, label in train_data:
        with autograd.record():
            out = net(data)
            loss = softmax_cross_entropy(out, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(out, label)
    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss : %f, Train acc : %f, Test acc : %f" % 
                (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))
    
data, label = mnist_test[0 : 19]
print("true labels")
print(get_text_labels(label))
predicted_labels = net(data).argmax(axis = 1)
print("predicted labels")
print(get_text_labels(predicted_labels.asnumpy()))