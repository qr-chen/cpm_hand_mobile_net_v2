# -*- coding:utf-8 -*-
import mxnet as mx
import mxnet.gluon as g
from mobile_net_v2 import *


def conv_2d(sub_model, channels, kernel_size, activation='relu'):
    '''
    定义一个用于实现same conv的helper function
    activation: 'relu' or None
    '''
    sub_model.add(nn.Conv2D(channels=channels, kernel_size=kernel_size,
                            strides=1, padding=int((kernel_size - 1) / 2), activation=activation))
    # sub_model.add(nn.BatchNorm()) # beta_initializer = 'zeros', gamma_initializer = 'ones'
    # sub_model.add(nn.Activation(activation))


def max_pooling(sub_model):
    sub_model.add(nn.MaxPool2D(strides=2))


class CPM(g.nn.HybridBlock):
    def __init__(self, stages, joints, width_mult=1):
        super(CPM, self).__init__()

        self.stages = stages
        self.stage_heatmap = []
        self.joints = joints
        self.w = width_mult

        self.first_oup = 32 * self.w
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, "stage0_"],  # -> 184x184
            [6, 24, 2, 2, "stage1_"],  # -> 92x92
            [6, 32, 3, 1, "stage2_"],  # -> 92x92  pre:[6, 32, 3, 2, "stage2_"]
            [6, 64, 4, 2, "stage3_0_"],  # -> 46x46
            [6, 96, 3, 1, "stage3_1_"],  # -> 46x46
            [6, 160, 3, 1, "stage4_0_"],  # -> 46x46  pre:[6, 160, 3, 2, "stage4_0_"]
            [6, 320, 1, 1, "stage4_1_"],  # -> 46x46
        ]
        self.last_channels = int(1280 * self.w) if self.w > 1.0 else 1280

        with self.name_scope():
            self.sub_stage = nn.HybridSequential('sub_stage')  # feature extractor part
            with self.sub_stage.name_scope():
                self.sub_stage = nn.HybridSequential()
                self.sub_stage.add(ConvBlock(self.first_oup, 3, 2, prefix="stage0_"))  # 368x368 -> 184x184
                inp = self.first_oup
                for t, c, n, s, prefix in self.interverted_residual_setting:
                    oup = c * self.w
                    self.sub_stage.add(ExpandedConvSequence(t, inp, oup, n, s, prefix=prefix))
                    inp = oup

                self.sub_stage.add(Conv1x1(self.last_channels, prefix="stage4_2_"))

            self.model_stage_1 = nn.HybridSequential('model_stage_1')
            with self.model_stage_1.name_scope():
                conv_2d(self.model_stage_1, channels=512, kernel_size=1)
                conv_2d(self.model_stage_1, channels=self.joints, kernel_size=1)

            self.model_stage_2 = nn.HybridSequential('stage2_')
            with self.model_stage_2.name_scope():
                # input
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=7)
                conv_2d(self.model_stage_2, channels=128, kernel_size=1)
                conv_2d(self.model_stage_2, channels=self.joints, kernel_size=1)
                # output append

            self.model_stage_3 = nn.HybridSequential('stage3_')
            with self.model_stage_3.name_scope():
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=7)
                conv_2d(self.model_stage_3, channels=128, kernel_size=1)
                conv_2d(self.model_stage_3, channels=self.joints, kernel_size=1)

            self.model_stage_4 = nn.HybridSequential('stage4_')
            with self.model_stage_4.name_scope():
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=7)
                conv_2d(self.model_stage_4, channels=128, kernel_size=1)
                conv_2d(self.model_stage_4, channels=self.joints, kernel_size=1)

            self.model_stage_5 = nn.HybridSequential('stage5_')
            with self.model_stage_5.name_scope():
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=7)
                conv_2d(self.model_stage_5, channels=128, kernel_size=1)
                conv_2d(self.model_stage_5, channels=self.joints, kernel_size=1)

            self.model_stage_6 = nn.HybridSequential('stage6_')
            with self.model_stage_6.name_scope():
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=7)
                conv_2d(self.model_stage_6, channels=128, kernel_size=1)
                conv_2d(self.model_stage_6, channels=self.joints, kernel_size=1)

    def forward(self, x):
        self.sub_stage_img_feature = self.sub_stage(x)
        self.stage1_heatmap = self.model_stage_1(self.sub_stage_img_feature)
        self.stage1_featuremap = mx.ndarray.concat(self.stage1_heatmap, self.sub_stage_img_feature)
        self.stage2_heatmap = self.model_stage_2(self.stage1_featuremap)
        self.stage2_featuremap = mx.ndarray.concat(self.stage2_heatmap, self.sub_stage_img_feature)
        self.stage3_heatmap = self.model_stage_3(self.stage2_featuremap)
        self.stage3_featuremap = mx.ndarray.concat(self.stage3_heatmap, self.sub_stage_img_feature)
        self.stage4_heatmap = self.model_stage_4(self.stage3_featuremap)
        self.stage4_featuremap = mx.ndarray.concat(self.stage4_heatmap, self.sub_stage_img_feature)
        self.stage5_heatmap = self.model_stage_5(self.stage4_featuremap)
        self.stage5_featuremap = mx.ndarray.concat(self.stage5_heatmap, self.sub_stage_img_feature)
        self.stage6_heatmap = self.model_stage_6(self.stage5_featuremap)
        self.totol_heatmap = mx.ndarray.concat(self.stage1_heatmap, self.stage2_heatmap, self.stage3_heatmap,
                                               self.stage4_heatmap, self.stage5_heatmap, self.stage6_heatmap)
        return self.totol_heatmap  # label contact 6次
