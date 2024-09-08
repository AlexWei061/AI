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
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))           # 把ID信息去掉 因为ID不会提供预测信息
# print(all_features)

"""数据预处理"""
