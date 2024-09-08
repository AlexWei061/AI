# House Price Prediction

&emsp; 这里用 $linear\;regression$ 的方式来搞一下预测房价问题，这是在 $kaggle$ 的一个比赛 不限期 可以随时提交。

## Step 1.  数据处理

&emsp; 关于数据的下载，直接在 $kaggle$ 网站上面点击 $download$ 就可以了...

&emsp; 然后就是数据的处理，首先下载下来的数据是两个 $csv$ 文件，于是我们考虑用 $python$ 中的 $pandas$ 进行数据的处理。

&emsp; 这里有一点 $pandas$ 的笔记qwq：[关于pandas](./note/关于pandas.md)

&emsp; 首先把下下来的两个文件 $kaggle\_house\_pred\_train.csv$ 和 $kaggle\_house\_pred\_test.csv$ 保存在 $data$ 文件夹里，然后在python里读取他们就是这样：

```python
test_data  = pd.read_csv('./data/kaggle_house_pred_test.csv')
train_data = pd.read_csv('./data/kaggle_house_pred_train.csv')
```

&emsp; 这样我们就得到了两个 $pandas$ 的 $DataFrame$。

&emsp; 我们可以看看这些特征都有些啥：

```python
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

```
输出：
   Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
0   1          60       RL         65.0       WD        Normal     208500
1   2          20       RL         80.0       WD        Normal     181500
2   3          60       RL         68.0       WD        Normal     223500
3   4          70       RL         60.0       WD       Abnorml     140000
```

&emsp; 我们会发现这些特种中有数字也有字符而且还有 $ID$ 这种对我们进行预测没用的编号信息，甚至还有缺失的信息，所以我们还需要对这些数据进行一些进一步的处理。

&emsp; 第一步就是把 $ID$ 给去掉：

```python
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

&emsp; 然后我们考虑将**所有缺失的特征都替换成对应特征的平均值**，然后我们为了将所有特征都放在一个共同的尺寸上，我们考虑**将特征都缩放到零均值和单位方差来标准化数据**，也就是（下面的 $\mu, \sigma$ 分别表示均值和标准差 ：

$$ x \rightarrow \frac{x - \mu}{\sigma} $$

&emsp; 于是我们这样：

```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index                 # 所有feature中类型不是'object'的值的下标
all_features[numeric_features] = all_features[numeric_features].apply(                        # 标准化数据 均值变成0

```

&emsp; 标准化数据后，均值变成$0$ 了，于是我们就给所有缺失的值填成 $0$：

```python
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

&emsp; 然后，我们用 $pandas$ 的 $get_dummies()$ 函数来将所有非数字的特征变成数字：

```python
all_features = pd.get_dummies(all_features, dummy_na=True)
```

&emsp; 最后一步，我们将 $pandas$ 的数据转化成 $numpy$ 的数据就可以开始训练了：

```python
n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values, dtype = np.float32)
test_features = np.array(all_features[n_train:].values, dtype = np.float32)
train_labels = np.array(train_data.SalePrice.values.reshape(-1, 1), dtype = np.float32)
```

## Step 2. 训练

&emsp; 虽然线性回归模型在比赛中估计拿不到什么好成绩，但是我最近在学这个，也就没办法了qwq ~~绝对不是我只会这个的原因哦\doge~~

&emsp; 首先我们的 $loss$ 函数，就直接用 $gluon$ 里的损失平方模型了：

```python
loss = gluon.loss.L2Loss()
```

&emsp; 然后我们定义一个一层的 $net$ 模型。

```python
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

&emsp; 然后我们看看关于如何评价我们训练出来的数据的准确度的问题，我们最先能想到的就是做差取平均这个方法了吧，但是这个方法有个很大的问题。举个例子来说，如果我们在俄亥俄州农村地区估计一栋房子的价格时，假设我们的预测偏差了10万美元，然而那里一栋典型的房子的价值是12.5万美元， 那么模型可能做得很糟糕，另一方面，如果我们在加州豪宅区的预测出现同样的10万美元的偏差，（在那里，房价中位数超过400万美元）这可能是一个不错的预测。

&emsp; 所以我们要关注的不是绝对数量，而是相对数量，因此我们更关心 $\frac{y - \hat{y}}{y}$ 而不是 $y - \hat{y}$。而事实上，$kaggle$ 的评测系统也给出了一种评判方法：

$$ \frac 1n\sqrt{\sum_{i = 1}^n\left( \log y_i - \log \hat{y_i} \right)^2} $$

&emsp; 也就是这样：

```python
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

&emsp; 然后就是训练的部分了，也是中规中矩：

```python
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
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
    # 这里返回每个epoch训练完之后的accurcy list然后train_ls[-1]和test_lr[-1]就是最终训练出来的效果
    return train_ls, test_ls
```

## Step 3. K折交叉验证

&emsp; 听着这个名字很高大上，其实就是调参的意思...

&emsp; 简单的来说就是把一些 $X, y$ 切片成 $k$ 份，然后把第 $i$ 份作为验证数据，而其他部分作为训练数据，我们首先写一个 $get\_data()$ 的函数用来得到 $k\_fold$ 的数据：

```python
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
```

&emsp; 然后我们就可以写 $k \_ fold()$ 了：

```python
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, ' f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k                    # 返回 k 折训练效果的平均值
```

&emsp; 然后我们就可以开始训练了：

```python
k, num_epochs, lr, weight_decay, batch_size = 10, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, ' f'平均验证log rmse: {float(valid_l):f}')
```

&emsp; 然后我们就能用这一段代码愉快地开始调参了，改一改 $k, lr, wd$ 啥的看看能不能把 $log \_ rmse$ 的均值降下去

## Step 4. 提交到Kaggle

&emsp; 既然参数已经调好了，那么我们就可以不用管 $k\_fold$ 了，直接用所有的训练数据来对我们的模型进行训练。并把训练出来的结果保存在 $submission.csv$ 文件中，然后把这个文件交给 $kaggle$ 测评就好了：

```python
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    preds = net(test_features).asnumpy()                                               # 将网络应用于测试集。
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])                        # 将其重新格式化以导出到Kaggle
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis = 1)
    submission.to_csv('submission.csv', index = False)
```

&emsp; 然后：

```python
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
```

&emsp; 就搞定了嘻嘻~~

&emsp; 完整代码：

```python
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
    # 这里返回每个epoch训练完之后的accurcy list然后train_ls[-1]和test_lr[-1]就是最终训练出来的效果
    return train_ls, test_ls



"""K折交叉验证 其实就是调参[合十]"""
def get_k_fold_data(k, i, X, y):                                                               # k折 第i份为测试数据 其余为训练数据
    assert k > 1
    fold_size = X.shape[0] // k                                                                # 分成 k 份
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part                                                  # valid 是测试数据
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
    preds = net(test_features).asnumpy()                                                       # 将网络应用于测试集。
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])                                # 将其重新格式化以导出到Kaggle
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
```