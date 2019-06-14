# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
import cPickle
from bbox.bbox_transform import bbox_poly2hbb, poly2bbox, polygonToRotRectangle_batch, choose_best_Rroi_batch

from core.rcnn import sample_Rrois
import copy
import pdb
DEBUG = False

# v2 is the version with Rroi elarge
class RRoITargetRotBox_v2Operator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction):
        super(RRoITargetRotBox_v2Operator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._cfg = cfg
        self._fg_fraction = fg_fraction

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois == -1 or self._batch_rois % self._batch_images == 0, \
            'batchimages {} must devide batch_rois {}'.format(self._batch_images, self._batch_rois)
        # pdb.set_trace()
        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()

        if self._batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_boxes.shape[0]
            fg_rois_per_image = rois_per_image
        else:
            rois_per_image = self._batch_rois / self._batch_images
            fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        # pdb.set_trace()
        gt_rotboxes = np.concatenate((polygonToRotRectangle_batch(gt_boxes[:, :-1]), gt_boxes[:, -1][:, np.newaxis]), axis=1).astype(np.float32)

        all_rois = np.vstack((all_rois, np.hstack((zeros, choose_best_Rroi_batch(gt_rotboxes[:, :-1])))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'
        gpu_id = in_data[0].context.device_id
        rois, labels, bbox_targets, bbox_weights = \
            sample_Rrois(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, self._cfg, gt_boxes=gt_rotboxes, device_id=gpu_id)

        # elarge roi for feature extraction
        # rois: (n, 6) (batch, x, y, w, h ,theta)
        # pdb.set_trace()
        elarge_rois = copy.deepcopy(rois)
        elarge_rois[:, 3] = rois[:, 3] * 1.2
        elarge_rois[:, 4] = rois[:, 4] * 1.4

        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))
        # pdb.set_trace()
        for ind, val in enumerate([rois, elarge_rois, labels, bbox_targets, bbox_weights]):
            self.assign(out_data[ind], req[ind], val)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register('RRoI_target_rotbox_v2')
class RRoITargetRotbox_v2Prop(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction='0.25'):
        super(RRoITargetRotbox_v2Prop, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._cfg = cPickle.loads(cfg)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['Rrois', 'gt_boxes']

    def list_outputs(self):
        return ['Rrois_output', 'Rrois_output_elarge', 'Rlabel', 'Rbbox_target', 'Rbbox_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        rois = rpn_rois_shape[0] + gt_boxes_shape[0] if self._batch_rois == -1 else self._batch_rois
        # rois = rpn_rois_shape[0] if self._batch_rois == -1 else self._batch_rois

        output_rois_shape = (rois, 6)
        label_shape = (rois, )
        bbox_target_shape = (rois, 5 * self._num_classes)
        bbox_weight_shape = (rois, 5 * self._num_classes)

        return [rpn_rois_shape, gt_boxes_shape], \
               [output_rois_shape, output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

        # return [rpn_rois_shape], \
        #        [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]


    def create_operator(self, ctx, shapes, dtypes):
        return RRoITargetRotBox_v2Operator(self._num_classes, self._batch_images, self._batch_rois, self._cfg, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
