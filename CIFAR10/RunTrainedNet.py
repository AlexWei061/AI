#coding = utf-8

import collections
import math
import os
import shutil
import pandas as pd
from mxnet import gluon, init, npx, autograd, nd
from mxnet.gluon import nn
from d2l import mxnet as d2l
import time

npx.set_np()

is_save, is_load = True, True
num_epoch, batch_size, lr, lr_decay, lr_period, wd = 1, 128, 0.02, 0.9, 4, 5e-4

transform_train = gluon.data.vision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    gluon.data.vision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64～1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    gluon.data.vision.transforms.RandomResizedCrop(32, scale = (0.64, 1.0),
                                                   ratio = (1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # 标准化图像的每个通道
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])

transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])

train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join('./data', 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]

train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle = True,
    last_batch = 'discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle = False,
    last_batch = 'discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle = False,
    last_batch = 'keep')

class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv = False, strides = 1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size = 3, padding = 1, strides = strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size = 3, padding = 1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size = 1, strides = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)

def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size = 3, strides = 1, padding = 1),
            nn.BatchNorm(),
            nn.Activation('relu'))
    
    def resnet_block(num_channels, num_residuals, first_block = False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv = True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block = True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net

ctx = d2l.try_gpu()

num_classes = 10
net = resnet18(num_classes)
if is_load:
    net.load_parameters('resnet_params.params')
else:
    net.initialize(ctx = ctx, init = init.Xavier())
net.hybridize()

preds = []

start_time = time.time()
for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(ctx))
    preds.extend(y_hat.argmax(axis = 1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index = False)
end_time = time.time()
print("Used time :", (end_time - start_time) / 60, "min")