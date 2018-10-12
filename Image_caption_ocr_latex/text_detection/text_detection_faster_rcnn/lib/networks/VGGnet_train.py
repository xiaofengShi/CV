'''
File: VGGnet_train.py
Project: networks
File Created: Monday, 2nd April 2018 12:11:51 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Wednesday, 20th June 2018 9:02:53 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''


import tensorflow as tf
from .config import cfg
from .network import Network


class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes,
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):
        # n_classes = 21
        n_classes = 3
        anchor_scales = [8]
        _feat_stride = [16, ]
        # print('n_classes', n_classes)
        # print('anchor_scales', anchor_scales)
        # net frame
        # base net vgg16
        (self.feed('data')
            .conv(3, 3, 64, 1, 1, name='conv1_1')
            .conv(3, 3, 64, 1, 1, name='conv1_2')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
            .conv(3, 3, 128, 1, 1, name='conv2_1')
            .conv(3, 3, 128, 1, 1, name='conv2_2')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
            .conv(3, 3, 256, 1, 1, name='conv3_1')
            .conv(3, 3, 256, 1, 1, name='conv3_2')
            .conv(3, 3, 256, 1, 1, name='conv3_3')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
            .conv(3, 3, 512, 1, 1, name='conv4_1')
            .conv(3, 3, 512, 1, 1, name='conv4_2')
            .conv(3, 3, 512, 1, 1, name='conv4_3')
            .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
            .conv(3, 3, 512, 1, 1, name='conv5_1')
            .conv(3, 3, 512, 1, 1, name='conv5_2')
            .conv(3, 3, 512, 1, 1, name='conv5_3'))
        # ========= RPN ============
        # 该层对上层的feature map进行卷积，生成512通道的的feature map
        (self.feed('conv5_3').conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))
        # 卷积最后一层的的feature_map尺寸为batch*h*w*512

        # 原来的单层双向LSTM
        # (self.feed('rpn_conv/3x3').Bilstm(512, 128, 512, name='lstm_o'))
        # bilstm之后输出的尺寸为(N, H, W, 512)

        # 现在改成的多层双向LSTM，添加droupout和LN
        (self.feed('rpn_conv/3x3').MutBigru(512, 128, 512, name='lstm_o'))
        # 使用LSTM的输出来计算位置偏移和类别概率（判断是否是物体，不判断类别的种类）
        # 默认将最后的图像对应到10个gride
        # 输入尺寸为(N, H, W, 512)  输出尺寸（N, H, W, int(d_o)）
        # 可以将这一层当做目标检测中的最后一层feature_map
        # 对于rpn_bbox_pred--对于h*w的尺寸上，每一个单位上生成4个位置偏移量
        # 对于rpn_cls_score--对于h*w的尺寸上，每一个单位上生成2个置信度得分，判断是否为物体
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * 10 * 2, name='rpn_cls_score'))

        # generating training labels on the fly
        # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
        # 给每个anchor上标签，并计算真值（也是delta的形式），以及内部权重和外部权重
        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
            .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))

        # ===========ROI PROPOSAL=========== #
        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
            .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
            .spatial_softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob').spatial_reshape_layer(len(anchor_scales) * 10 * 2, name='rpn_cls_prob_reshape'))

        self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info').proposal_layer(
            _feat_stride, anchor_scales, 'TRAIN', name='rpn_rois')

        (self.feed('rpn_rois', 'gt_boxes').proposal_target_layer(n_classes, name='roi-data'))

        # ========= RCNN ============
        (self.feed('conv5_3', 'roi-data').roi_pool(7, 7, 1.0/16, name='pool_5')
             .fc(4096, name='fc6').dropout(0.5, name='drop6')
             .fc(4096, name='fc7').dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score').softmax(name='cls_prob'))

        (self.feed('drop7').fc(n_classes*4, relu=False, name='bbox_pred'))
