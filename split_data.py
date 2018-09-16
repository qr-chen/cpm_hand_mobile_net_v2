# -*- coding:utf-8 -*-
"""
功能：从打的数据集中划分一部分数据作为测试集
DATA_PATH：总体数据集的路径
TEST_DATA_PATH：测试数据集的路径

"""

from sklearn.model_selection import train_test_split
import os

DATA_PATH = './train_dataset/'
TEST_DATA_PATH = './val_dataset/'

file_list = os.listdir(DATA_PATH)
img_list = [i for i in file_list if i.endswith('.jpg')]
json_list = [i for i in file_list if i not in img_list]
img_train, img_test, json_train, json_test = train_test_split(img_list, json_list, test_size=0.03)  # 可设置测试集比例

for im in img_test:
    os.rename(DATA_PATH + im, TEST_DATA_PATH + im)
for json in json_test:
    os.rename(DATA_PATH + json, TEST_DATA_PATH + json)

print 'done'