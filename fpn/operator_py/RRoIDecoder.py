# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Haozhi Qi
# --------------------------------------------------------

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

from bbox.bbox_transform import bbox_pred, clip_boxes
from rpn.generate_anchor import generate_anchors
from nms.nms import gpu_nms_wrapper
# from poly_nms_gpu.poly_nms import poly_gpu_nms
from dota_kit.poly_nm   s_gpu.nms import poly_gpu_nms_wrapper
from bbox.bbox_transform import dbbox_transform2_inv_warp
from bbox.bbox_transform import clip_polys, RotBox2Polys, polygonToRotRectangle_batch, choose_best_Rroi_batch
import cPickle
import pdb
import copy
DEBUG = False

## version 2 did not apply nms
class RRoIDecoderOperator(mx.operator.CustomOp):
    def __init__(self, pre_nms_top_n, post_nms_top_n, threshold, min_size, cfg):
        super(RRoIDecoderOperator, self).__init__()
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self._threshold = threshold
        self._min_size = min_size
        self._cfg = cfg

    def forward(self, is_train, req, in_data, out_data, aux):
        # batch_size = in_data[0].shape[0]
        batch_size = in_data[0][0][0]
        if batch_size.asnumpy() > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        rois = in_data[0].asnumpy()
        st_pred = in_data[1].asnumpy()
        # st_score: shape (n, 2)
        st_score = in_data[2].asnumpy()[:, :, 1].reshape(-1, 1)
        im_info = in_data[-1].asnumpy()[0, :]

        pre_nms_topN = self._pre_nms_top_n
        post_nms_topN = self._post_nms_top_n
        min_size = self._min_size
        # 1. generate Rrois
        cfg = self._cfg
        # checked it, yes, the weights is different in training and testing, so the st_pred is different in training and testing
        # this is very critical
        if is_train:
            if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
                means = np.tile(np.array(cfg.TRAIN.BBOX_MEANS), 2 if cfg.CLASS_AGNOSTIC else cfg.dataset.NUM_CLASSES)
                stds = np.tile(np.array(cfg.TRAIN.BBOX_STDS), 2 if cfg.CLASS_AGNOSTIC else cfg.dataset.NUM_CLASSES)
                st_pred = st_pred * stds + means
        Rrois = dbbox_transform2_inv_warp(rois[:, 1:], st_pred)[:, 5:]
        if (len(Rrois) == 0):
            pdb.set_trace()
        # remove Rrois with either height or width < thredhold
        keep = self._filter_boxes_v2(Rrois, min_size * im_info[2] * min_size * im_info[2])
        keep_Rrois = Rrois[keep]
        scores = st_score[keep]

        if len(keep_Rrois) == 0:
            Rrois[:, 2] = np.maximum(Rrois[:, 2], min_size * im_info[2])
            Rrois[:, 3] = np.maximum(Rrois[:, 3], min_size * im_info[2])
            # if after filter, there are no instances, clip all Rrois' size
            keep_Rrois = Rrois
            scores = st_score
        proposals = RotBox2Polys(keep_Rrois)

        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        # take after_nms_topN (e.g. 300)
        # return the top proposals (-> RoIs top)
        det = np.hstack((proposals, scores)).astype(np.float32)

        keep = np.arange(len(det))
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        # pad to ensure output size remains unchanged
        if len(keep) < post_nms_topN:
            pad = npr.choice(keep, size=post_nms_topN - len(keep))
            keep = np.hstack((keep, pad))
        proposals = proposals[keep, :]

        scores = scores[keep]
        # -----------------------------
        # trans polys to rotboxes
        proposals = polygonToRotRectangle_batch(proposals)
        # range angle in [0, 180] to eliminate ambiguity of orientation agnostic instance regression
        proposals = choose_best_Rroi_batch(proposals)
        # proposals: (x_ctr, y_ctr, w, h, angle)
        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        # if is_train:
        self.assign(out_data[0], req[0], blob)

        # elarged area for feature extraction
        elarge_proposals = copy.deepcopy(proposals)
        elarge_proposals[:, 2] = proposals[:, 2] * 1.2
        elarge_proposals[:, 3] = proposals[:, 3] * 1.4
        elarge_blob = np.hstack((batch_inds, elarge_proposals.astype(np.float32, copy=False)))
        self.assign(out_data[1], req[1], elarge_blob)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2]
        hs = boxes[:, 3]
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _filter_boxes_v2(boxes, area):
        """ Remove all boxes with area below 10 * 10 """
        ws = boxes[:, 2]
        hs = boxes[:, 3]
        # keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        keep = np.where(ws * hs >= area)[0]
        return keep

@mx.operator.register("RRoIDecoder")
class RRoIDecoderProp(mx.operator.CustomOpProp):
    def __init__(self, cfg, Rroi_pre_nms_top_n='12000', Rroi_post_nms_top_n='2000', threshold='0.5', min_size='10'):
        super(RRoIDecoderProp, self).__init__(need_top_grad=False)
        self._cfg = cPickle.loads(cfg)
        self._Rroi_pre_nms_top_n = int(Rroi_pre_nms_top_n)
        self._Rroi_post_nms_top_n = int(Rroi_post_nms_top_n)
        self._threshold = float(threshold)
        self._min_size = int(min_size)

    def list_arguments(self):

        return ['rois', 'bbox_pred', 'cls_prob', 'im_info']

    def list_outputs(self):

        # return ['output_rois', 'output_rois_L']
        return ['output', 'output_rois_L']

    def infer_shape(self, in_shape):
        output_shape = (self._Rroi_post_nms_top_n, 6)

        return in_shape, [output_shape, output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return RRoIDecoderOperator(self._Rroi_pre_nms_top_n, self._Rroi_post_nms_top_n,
                                       self._threshold, self._min_size, self._cfg)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
