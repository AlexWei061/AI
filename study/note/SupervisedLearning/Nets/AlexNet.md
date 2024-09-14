# Alex Net

&emsp; ~~作者的名字也叫Alex哦 (0\^w\^0)~~

>2012年，AlexNet横空出世。它首次证明了学习到的特征可以超越手工设计的特征。它一举打破了计算机视觉研究的现状。 AlexNet使用了8层卷积神经网络，并以很大的优势赢得了2012年ImageNet图像识别挑战赛。

>AlexNet和LeNet的架构非常相似，如 :numref:fig_alexnet所示。 注意，本书在这里提供的是一个稍微精简版本的AlexNet，去除了当年需要两个小型GPU同时运算的设计特点。

>AlexNet和LeNet的设计理念非常相似，但也存在显著差异。

>AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层。

>AlexNet使用ReLU而不是sigmoid作为其激活函数。下面的内容将深入研究AlexNet的细节。

## 定义

&emsp; 这个 $net$ 其实就是比 $LeNet$ 多了几层而已，就像这样：

```python
"""Alex Net"""
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
```

&emsp; 我们可以看看打印出来的 $net$ 长啥样：

```
Sequential(
  (0): Conv2D(None -> 96, kernel_size=(11, 11), stride=(4, 4), Activation(relu))
  (1): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
  (2): Conv2D(None -> 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), Activation(relu))
  (3): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
  (4): Conv2D(None -> 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), Activation(relu))
  (5): Conv2D(None -> 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), Activation(relu))
  (6): Conv2D(None -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), Activation(relu))
  (7): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
  (8): Flatten
  (9): Dense(None -> 4096, Activation(relu))
  (10): Dropout(p = 0.5, axes=())
  (11): Dense(None -> 4096, Activation(relu))
  (12): Dropout(p = 0.5, axes=())
  (13): Dense(None -> 10, linear)
)
```

&emsp; 然后我把 $LeNet$ 的放在下面，可以对比一下：

```python
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(channels = 20, kernel_size = 5, activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 2, strides = 2))
    net.add(nn.Conv2D(channels = 50, kernel_size = 3, activation = 'relu'))
    net.add(nn.MaxPool2D(pool_size = 2, strides = 2))
    net.add(nn.Flatten())
    net.add(nn.Dense(128, activation = 'relu'))
    net.add(nn.Dense(10))
```

&emsp; 感觉其实没啥区别 ~~虽然学术界LeNet到AlexNet花了20年~~，论文里面似乎有一些高大上的解释，但是我感觉其实是先有了这个 $net$ 它work，才会有人去解释它为什么work \doge。

&emsp; 这个 $AlexNet$ 跑起来就很慢了，于是我们在后面的计算中尝试一下使用gpu来运算。

## 训练

&emsp; 没什么好说的，和 $LeNet$ 的训练没啥区别，直接上代码。

&emsp; 唯一值得注意的就是，一开始的 $AlexNet$ 是跑的 $Imagenet$ 上的数据，而我们这里训练时还是用的 $mnist$ 所以我们需要把 $mnist$ 中 $28 \times 28$ 的 图片resize成 $Imagenet$ 的 $224 \times 224$ 。

```python
batch_size, num_epoch, lr = 128, 5, 0.01
train_data, test_data = d2l.load_data_fashion_mnist(batch_size, resize = 224)                  # resize from 28 * 28 to 224 * 224

ctx = d2l.try_gpu()
net.initialize(ctx = ctx, init = init.Xavier())                                                # 一个particular的初始化函数

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
```

&emsp; 让我们看看跑出来的效果吧：~~（笔者gpu环境还没配好，这一段玩意儿跑了特别久特别久，一个epoch就好几分钟~~

```
Epoch 0. Loss : 1.293490, Train acc : 0.516247, Test acc : 0.756329
Epoch 1. Loss : 0.656114, Train acc : 0.756191, Test acc : 0.804193
Epoch 2. Loss : 0.540371, Train acc : 0.797880, Test acc : 0.835443
Epoch 3. Loss : 0.478858, Train acc : 0.823977, Test acc : 0.854233
Epoch 4. Loss : 0.433456, Train acc : 0.842173, Test acc : 0.866199
```