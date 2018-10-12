import sys
import os
import cv2
import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(__file__))
from ctpn.detectors import TextDetector
from ctpn.model import ctpn, ctpn_ori
from ctpn.other import draw_boxes
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg

'''
load network
输入的名称为'Net_model'
'VGGnet_test'--test
'VGGnet_train'-train
'''


def load_tf_model(ckpt_text_formula_path, sess2, g2_text_formula):
    with sess2.as_default():
        with g2_text_formula.as_default():
            cfg.TEST.HAS_RPN = True  # Use RPN for proposals
            net = get_network("VGGnet_test")
            saver_ctpn = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(ckpt_text_formula_path)
            reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print("Tensor_name is : ", key)
                # print(reader.get_tensor(key))
            saver_ctpn.restore(sess2, ckpt.model_checkpoint_path)
            print("Load text and formula detection net done")
    return sess2, saver_ctpn, net


def text_detect(img, sess, net, img_name, image_predict_path):
    # ctpn网络测到
    img = cv2.imread(img)
    scores, boxes, img = ctpn(img, sess, net)
    textdetector = TextDetector()
    print('detect the box')
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    # text_recs, tmp = draw_boxes(img, boxes, caption='im_name', wait=True, is_display=False)
    text_rect_dict, tmp = draw_boxes(
        img, boxes, img_name, image_predict_path, caption='im_name', wait=True, is_display=True)
    return text_rect_dict, tmp, img


""" 直接进行模型的测试 """
image_predict_path = '/Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/img_text_formula'


def text_detect_ori(img_path):
    global image_predict_path
    print('image_predict_path', image_predict_path)
    # ctpn网络测到
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    print(img_name)
    scores, boxes, img = ctpn_ori(img)
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    # text_recs, tmp = draw_boxes(img, boxes, caption='im_name', wait=True, is_display=False)
    text_recs, tmp = draw_boxes(
        img, boxes, img_name, image_predict_path, caption='im_name', wait=True, is_display=True)
    return text_recs, tmp, img


if __name__ == '__main__':
    text_recs, tmp, img = text_detect_ori(
        '/Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/text_formula_faster_rcnn/data/2.jpg')
    print(text_recs)
