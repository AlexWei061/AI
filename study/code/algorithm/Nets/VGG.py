from mxnet.gluon import nn
from mxnet import image
from d2l import mxnet as d2l
from mxnet import autograd, gluon, nd, init

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    with out.name_scope():
        for _ in range(num_convs):
            out.add(nn.Conv2D(channels = channels, kernel_size = 3, padding = 1, activation = 'relu'))
        out.add(nn.MaxPool2D(pool_size = 2, strides = 2))
    return out

# blk = vgg_block(2, 128)
# blk.initialize()
# x = nd.random.uniform(shape = (2, 3, 16, 16))                                                    # (batch_size, channels, height, width)
# print(blk(x).shape)

def vgg_stack(architecture):
    out = nn.Sequential()
    with out.name_scope():
        for (num_convs, channels) in architecture:
            out.add(vgg_block(num_convs, channels))
    return out

architecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = nn.Sequential()
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(4096, activation = 'relu'))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(4096, activation = 'relu'))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(10))

print(net)

batch_size, num_epoch, lr = 128, 5, 0.01
train_data, test_data = d2l.load_data_fashion_mnist(batch_size, resize = 96)                  # resize from 28 * 28 to 224 * 224

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