# coding = utf-8
import pandas as pd
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()

train_data = pd.read_csv('./data/kaggle_house_pred_tarin.csv')
test_data  = pd.read_csv('./data/kaggle_house_pred_test.csv')
print(train_data)
print(test_data)

