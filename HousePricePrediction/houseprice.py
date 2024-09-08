# coding = utf-8

import pandas as pd
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()

"""读取数据"""
train_data = pd.read_csv('./data/kaggle_house_pred_train.csv')
test_data  = pd.read_csv('./data/kaggle_house_pred_test.csv')
# print(train_data)
# print(test_data)
# print(train_data.shape)
# print(test_data.shape)
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))                   # 把ID信息去掉 因为ID不会提供预测信息
# print(all_features)

"""
数据预处理
1. 通过将特征重新缩放到零均值和单位方差来标准化数据
2. 将所有缺失的值替换为相应特征的平均值
"""
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index                 # 所有feature中类型不是'object'的值的下标
# print(numeric_features)
all_features[numeric_features] = all_features[numeric_features].apply(                        # 标准化数据 均值变成0
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)                     # 在标准化数据之后 将缺失值设置为0
all_features = pd.get_dummies(all_features, dummy_na = True)                                  # 将非数值量转化为数值
# print(all_features)
# print(all_features.shape)
n_train = train_data.shape[0]                                                                 # 将pandas转成numpy
train_features = np.array(all_features[:n_train].values, dtype = np.float32)
test_features  = np.array(all_features[n_train:].values, dtype = np.float32)
train_labels = np.array(train_data.SalePrice.values.reshape(-1, 1), dtype = np.float32)



"""训练"""
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

def log_rmse(net, features, labels):
    clipped_preds = np.clip(net(features), 1, float('inf'))                                    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())

def train(net, train_features, train_labels, test_features, test_labels,num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls



"""K折交叉验证 其实就是调参[合十]"""
def get_k_fold_data(k, i, X, y):                                 # k折 第i份为测试数据 其余为训练数据
    assert k > 1
    fold_size = X.shape[0] // k                                  # 分成 k 份
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part                    # valid 是测试数据
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate([X_train, X_part], 0)
            y_train = np.concatenate([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, ' f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k                      # 返回 k 折训练效果的平均值

k, num_epochs, lr, weight_decay, batch_size = 10, 300, 3, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, ' f'平均验证log rmse: {float(valid_l):f}')



"""开始训练"""
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    preds = net(test_features).asnumpy()                                                 # 将网络应用于测试集。
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])                          # 将其重新格式化以导出到Kaggle
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)