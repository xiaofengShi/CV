# coding:utf-8
import sys

import numpy as np

from .cfg import Config as cfg
from .other import normalize

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

from lib.fast_rcnn.nms_wrapper import nms
# from lib.fast_rcnn.test import  test_ctpn

from .text_proposal_connector import TextProposalConnector

DEBUG = False

CLASS = ('__background__', 'text', 'formula')


class TextDetector:
    """
        Detect text from an image
    """

    def __init__(self):
        """
        pass
        """
        self.text_proposal_connector = TextProposalConnector()

    def detect(self, text_proposals, scores, size):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        """
        # text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        if DEBUG:
            print('scores_shape', np.shape(scores))
            print('proposal_shape', np.shape(text_proposals))
        text_lines_dict = dict()
        for cls_ind, cls_name in enumerate(CLASS[1:]):
            cls_ind += 1
            cls_box = text_proposals[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, :, cls_ind]
            if DEBUG:
                print('===============================')
                print('cls_name:', cls_name)
                print('cls_scores_shape', np.shape(cls_scores))
            # 首先去掉当前类别小于设定阈值的box和score
            keep_inds = np.where(cls_scores > cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
            cls_box, cls_scores = cls_box[keep_inds], cls_scores[keep_inds]
            # 对去除之后的score进行排序
            sorted_indices = np.argsort(cls_scores.ravel())[::-1]
            if DEBUG:
                print('keep_inds:', np.shape(keep_inds))
            cls_box, cls_scores = cls_box[sorted_indices], cls_scores[sorted_indices]
            # 进行nums处理，去除重复框
            keep_inds = nms(np.hstack((cls_box, cls_scores)), cfg.TEXT_PROPOSALS_NMS_THRESH)
            cls_box, cls_scores = cls_box[keep_inds], cls_scores[keep_inds]
            cls_scores = normalize(cls_scores)
            if DEBUG:
                print('norml_socres', np.shape(cls_scores))
                print('size:', size)
                print('text_propoasal', np.shape(cls_box))
            text_lines = self.text_proposal_connector.get_text_lines(cls_box, cls_scores, size)

            keep_inds = self.filter_boxes(text_lines)
            text_lines = text_lines[keep_inds]
            if text_lines.shape[0] != 0:
                keep_inds = nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
                text_lines = text_lines[keep_inds]
                text_lines_dict[cls_name] = text_lines
        return text_lines_dict

    def filter_boxes(self, boxes):
        heights = boxes[:, 3] - boxes[:, 1] + 1
        widths = boxes[:, 2] - boxes[:, 0] + 1
        scores = boxes[:, -1]
        return np.where((widths / heights > cfg.MIN_RATIO) & (scores > cfg.LINE_MIN_SCORE) &
                        (widths > (cfg.TEXT_PROPOSALS_WIDTH * cfg.MIN_NUM_PROPOSALS)))[0]
