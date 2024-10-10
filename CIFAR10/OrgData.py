#coding = utf-8

import collections
import math
import os
import shutil
import pandas as pd
from mxnet import gluon, init, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
import time

npx.set_np()

def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

# labels = read_csv_labels(os.path.join('./data', 'trainLabels.csv'))
# print('# 训练样本 :', len(labels))
# print('# 类别 :', len(set(labels.values())))

def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))
        
def reorg_cifar10_data(data_dir, valid_ratio):
    start_time = time.time()
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    end_time = time.time()
    print("Finished read_csv_labels, Used time : %f seconds" % (end_time - start_time))
    start_time = time.time()
    reorg_train_valid(data_dir, labels, valid_ratio)
    end_time = time.time()
    print("Finished reorg_train_valid, Used time : %f min" % ((end_time - start_time) / 60))
    start_time = time.time()
    reorg_test(data_dir)
    end_time = time.time()
    print("Finished reorg_test, Used time : %f min" % ((end_time - start_time) / 60))

batch_size = 128
valid_ratio = 0.1
reorg_cifar10_data('./data', valid_ratio)