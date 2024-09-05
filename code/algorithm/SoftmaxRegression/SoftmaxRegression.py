from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
from mxnet import nd
from mxnet import autograd

def transform(data, label):                                                              # 把图片转成矩阵的函数
    return data.astype('float32')/255, label.astype('float32')
mnist_train = gluon.data.vision.FashionMNIST(train = True, transform = transform)
mnist_test  = gluon.data.vision.FashionMNIST(train = True, transform = transform)

# 看看dataset下载成功与否
# data, label = mnist_train[0]
# print(data.shape)                                                                      # 一个3d的矩阵 (28, 28, 1)
# print(label)

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize = (15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = [ 't-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot' ]
    return [text_labels[int(i)] for i in label]

# 试试show_images的效果
# data, label = mnist_train[0 : 9]
# print(get_text_labels(label))
# show_images(data)

# Load 数据
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle = True)
test_data  = gluon.data.DataLoader(mnist_test,  batch_size, shuffle = False)

num_inputs  = 784                                                                # 28 * 28 = 784
num_outputs = 10                                                                 # 数据集有10中类型
w = nd.random_normal(shape = (num_inputs, num_outputs))                          # 随机初始化 w 和 b
b = nd.random_normal(shape = num_outputs)
params = [w, b]

for param in params:                                                             # 对每个param开一个梯度
    param.attach_grad()

def softmax(X):
    exp = nd.exp(X)                                                              # e^x 保证都是正数
    partition = exp.sum(axis = 1, keepdims = True)                               # 算行的和
    return exp / partition                                                       # 保证每行之和 = 1

# 试试 softmax 的效果
# X = nd.random_normal(shape = (2, 5))
# x_prob = softmax(X)
# print(X)
# print(x_prob)
# print(x_prob.sum(axis = 1))

def net(X):                                                                      # 逻辑回归方程 logistic regression
    return softmax(nd.dot(X.reshape((-1, num_inputs)), w) + b)

def cross_entropy(yhat, y):                                                      # 交叉熵损失函数
    return -nd.log(nd.pick(yhat, y)) # -nd.pick(nd.log(yhat), y)

def accuracy(output, label):                                                     # 计算拟合的准确度
    return nd.mean(output.argmax(axis = 1) == label).asscalar()                  # argmax是把每一列的最大概率的index返回出来 然后和label比较是否相同 最后所有求个mean

def evaluate_accuracy(data_itetator, net):
    acc = 0.
    for data, label in data_itetator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_itetator)

# 看看evaluate_accuracy的效果
# print(evaluate_accuracy(test_data, net))

learning_rate = .1
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

for epoch in range(10):
    train_loss = 0.
    train_acc  = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate / batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc  += accuracy(output, label)
    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss : %f, Train acc : %f, Test acc : %f" % (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))

# print(w)
# print(b)

data, label = mnist_test[0 : 9]
print("true labels")
print(get_text_labels(label))
predicted_labels = net(data).argmax(axis = 1)
print("predicted labels")
print(get_text_labels(predicted_labels.asnumpy()))
show_images(data)