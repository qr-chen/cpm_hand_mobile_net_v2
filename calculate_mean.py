# -*- coding:utf-8 -*-
import os
import numpy as np
from scipy.misc import imread
import glob
from tqdm import tqdm

IMGPATH = './train_dataset/'  # 数据集目录,只包含图片和标记
IMG_JSON = os.listdir(IMGPATH)

R_channel = 0
G_channel = 0
B_channel = 0

for file in tqdm(glob.glob(IMGPATH + '*.jpg')):
    img = imread(file)
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

num = len(IMG_JSON) / 2 * 368 * 368  # 这里（368,368）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
