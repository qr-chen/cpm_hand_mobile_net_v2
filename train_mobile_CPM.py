# -*- coding:utf-8 -*-
import re
# from mobile_CPM_model import *
from mobile_adjust_CPM_model import *
from multiprocessing import cpu_count
from model_utils import *
from mxnet.gluon import utils as gutils
from dataloder_mobile_cpm import *

epoch_loss = {}
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-4
lr_sch = mx.lr_scheduler.FactorScheduler(step=10000, factor=0.9)
CTX = [mx.gpu(0), mx.gpu(1)]
FREEZE_TOP_LAYERS = 0
TRAIN_DATA_ROOT = './train_dataset/'
VISUALIZE = True  # 可视化输出标记

train_dataset = ImageWithHeatMapDataset(root=TRAIN_DATA_ROOT)
train_data_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=cpu_count(), last_batch='rollover')  # cpu_count()

model = CPM_v2(stages=6, joints=21)
model.hybridize()
print('loading parameters and initializing model......')
model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=CTX)  # 初始化网络
# model.load_params('./model_saved/mobile_heatmap_final.params', ctx=CTX)
if FREEZE_TOP_LAYERS != 0:
    for layer_name, layer_params in model.collect_params().items():
        if 'sub_stage' in layer_name:
            if int(re.findall(r"\d", layer_name)[-1]) < FREEZE_TOP_LAYERS:
                layer_params.grad_req = 'null'

optimizer = g.Trainer(model.collect_params(), 'adam',
                      {'learning_rate': LEARNING_RATE, 'wd': 1e-5, 'lr_scheduler': lr_sch})
for epoch in range(EPOCHS):
    loss_list = []
    num_iters = 0
    for img, label in train_data_loader:
        num_iters += 1
        gpu_imgs = gutils.split_and_load(img, CTX)
        gpu_labels = gutils.split_and_load(label, CTX)
        gpu_labels[0] = gpu_labels[0]
        gpu_labels[1] = gpu_labels[1]

        # ===================forward=====================
        with mx.autograd.record():
            outputs = [model(img) for img in gpu_imgs]
            outputs[0] = outputs[0]
            outputs[1] = outputs[1]
            loss = [my_l2_loss(output, label) for output, label in zip(outputs, gpu_labels)]
        # ===================Visualize====================
        if VISUALIZE:
            if num_iters % 200 == 0:
                print('Starting Visualizing.......')
                save_stage_output_3(gpu_imgs, outputs, gpu_labels, num_iters, epoch)
        # ===================backward====================
        for l in loss:
            l.backward()
        optimizer.step(BATCH_SIZE)
        mx.nd.waitall()
        print ('Batch AVG loss is: %f ----- in epoch %d' % ((sum(loss)[0].asscalar()) / len(loss), epoch + 1))
        loss_list.append((sum(loss).asscalar()) / len(loss))

    # ===================save model and log========================
    if ((epoch + 1) % 50 == 0 and (epoch + 1) != EPOCHS):
        model.save_params('./model_saved/mobile_heatmap_%d.params' % (epoch + 1))
        # pickle_NDArray({'gpu_labels': gpu_labels, 'outputs': outputs,'img':gpu_imgs}, 'new_heatmap_dct%d.pkl' %(epoch+1))  # 保存输出数据

    epoch_mean_loss = np.mean(loss_list)
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, EPOCHS, epoch_mean_loss))
    epoch_loss['epoch_%d' % (epoch + 1)] = float(epoch_mean_loss)

model.save_params('./model_saved/mobile_heatmap_final.params')
with open('./model_saved/loss_log.json', 'w') as js_f:
    js_f.write(json.dumps(epoch_loss))
