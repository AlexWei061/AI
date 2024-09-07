from mxnet import autograd, gluon, np, npx, init
from d2l import mxnet as d2l
from mxnet.gluon import nn              # nn 是 neural network 的缩写

npx.set_np()

times = 5
batch_size = 10
training_size = 1000
true_w = np.array([3.4, -5.2])
true_b = 4
xs, ys = d2l.synthetic_data(true_w, true_b, training_size)

def load_array(data_arrays, batch_size, is_train = True): 
    # 构造一个Gluon数据迭代器
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle = is_train)
data_iter = load_array((xs, ys), batch_size)

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma = 0.01))
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

for i in range(times):
    for x, y in data_iter:
        with autograd.record():
            J = loss(net(x), y)
        J.backward()
        trainer.step(batch_size)
    trainedJ = loss(net(xs), ys)
    print("第 ", i + 1, " 次训练 : cost = ", trainedJ.mean())

w = net[0].weight.data()
b = net[0].bias.data()
print("true w : ", true_w)
print("true b : ", true_b)
print("w : ", w.reshape(1, 2))
print("b : ", b)