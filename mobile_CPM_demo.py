# -*- coding:utf-8 -*-
import os
# from mobile_CPM_model import *
from mobile_adjust_CPM_model import *
from model_utils import *

IMG_ROOT = './val_dataset/'
SAVE_ROOT = '/home/users/qiangrui.chen/results/'
CTX = mx.gpu(1)

model = CPM_v2(stages=6, joints=21)
model.hybridize()
# model.load_params('./model_initial/cpm_initial.params', ctx=CTX)
model.load_params('./model_saved/mobile_heatmap_final.params', ctx=CTX)  # 加载参数


def is_img(file_name):
    return file_name.endswith('.jpg')


IMG_LST = filter(is_img,  os.listdir(IMG_ROOT))
for image in IMG_LST:
    IMAGE_MEAN = nd.array([130.972915, 104.865190, 94.727472]).reshape(1, 1, 3)
    img = mx.image.imread(IMG_ROOT + image)
    img_resize = mx.image.imresize(img, 368, 368)
    img = nd.transpose(img_resize.astype('float32') - IMAGE_MEAN, (2, 0, 1))
    img = nd.expand_dims(img, axis=0)
    img = img.as_in_context(CTX)  # 数据派给GPU
    output = model(img)  # 修改img 通道
    # output = np.squeeze(output[:, -21:, :, :])
    output = np.squeeze(output[:, :21, :, :])

    plt.clf()
    plt.imshow(img_resize.asnumpy())

    cnt = 0
    location_list = []
    for joints in output:
        joints = joints.asnumpy()
        location = np.where(joints == np.max(joints))
        location_list.append(location)
        plt.plot(location[1][0] * 8, location[0][0] * 8, 'r.')  # 第二个索引目的是，如果有多个最大值，取第一次出现位置
        plt.text(location[1][0] * 8, location[0][0] * 8, '{0}'.format(cnt))  # 关节点序号
        cnt += 1

    plt.savefig(SAVE_ROOT + image)
    # plt.show()
    print 'successfully saved '
print 'done'
