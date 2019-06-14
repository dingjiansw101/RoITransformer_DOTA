# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Haozhi Qi, Yuwen Xiong
# --------------------------------------------------------

import mxnet as mx
import numpy as np
from mxnet.contrib import autograd
import gc
import pdb

class FPNRotatedROIAlignOperator(mx.operator.CustomOp):
    def __init__(self, feat_strides, pooled_height, pooled_width, output_dim):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.feat_strides = feat_strides
        self.output_dim = output_dim
        self.in_grad_hist_list = []
        self.num_strides = len(self.feat_strides)
        self.roi_pool = [None for _ in range(self.num_strides)]
        self.feat_idx = [None for _ in range(self.num_strides)]

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[-1].asnumpy()
        # w = rois[:, 3] - rois[:, 1] + 1
        # h = rois[:, 4] - rois[:, 2] + 1
        w = np.maximum(rois[:, 3], 1)
        h = np.maximum(rois[:, 4], 1)
        # TODO: carefully scale the w, h
        feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w * h) / 224)), 0, len(self.feat_strides) - 1)
        pyramid_idx = []

        rois_p = [None for _ in range(self.num_strides)]
        for i in range(self.num_strides):
            self.feat_idx[i] = np.where(feat_id == i)[0]
            if len(self.feat_idx[i]) == 0:
                # padding dummy roi
                rois_p[i] = np.zeros((1, 6))
                pyramid_idx.append(-1)
            else:
                rois_p[i] = rois[self.feat_idx[i]]
                pyramid_idx.append(self.feat_idx[i])
        rois_idx = np.argsort(np.hstack(pyramid_idx))[-rois.shape[0]:]

        if is_train:
            for i in range(self.num_strides):
                self.in_grad_hist_list.append(mx.nd.zeros_like(in_data[i]))


            autograd.mark_variables([in_data[i] for i in range(self.num_strides)], self.in_grad_hist_list)
            with autograd.train_section():
                for i in range(self.num_strides):
                    self.roi_pool[i] = mx.nd.contrib.ROIAlignRotated(in_data[i], mx.nd.array(rois_p[i], in_data[i].context), (7, 7), spatial_scale=1.0/self.feat_strides[i], sample_ratio=4)
                    # self.roi_pool[i] = mx.nd.ROIPooling(in_data[i], mx.nd.array(rois_p[i], in_data[i].context), (7, 7), spatial_scale=1.0 / self.feat_strides[i])

            roi_pool = mx.nd.concatenate(self.roi_pool, axis=0)
        else:
            # during testing, there is no need to record variable, thus saving memory
            # pdb.set_trace()
            roi_pool = [None for _ in range(self.num_strides)]
            for i in range(self.num_strides):
                # roi_pool[i] = mx.nd.ROIPooling(in_data[i], mx.nd.array(rois_p[i], in_data[i].context), (7, 7), spatial_scale=1.0 / self.feat_strides[i])
                roi_pool[i] = mx.nd.contrib.ROIAlignRotated(in_data[i], mx.nd.array(rois_p[i], in_data[i].context),
                                                                 (7, 7), spatial_scale=1.0 / self.feat_strides[i],
                                                                 sample_ratio=4)
            roi_pool = mx.nd.concatenate(roi_pool, axis=0)
        # pdb.set_trace()
        roi_pool = mx.nd.take(roi_pool, mx.nd.array(rois_idx, roi_pool.context))
        self.assign(out_data[0], req[0], roi_pool)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

        with autograd.train_section():
            for i in range(self.num_strides):
                if len(self.feat_idx[i] > 0):
                    autograd.compute_gradient([mx.nd.take(out_grad[0], mx.nd.array(self.feat_idx[i], out_grad[0].context)) * self.roi_pool[i]])


        for i in range(0, self.num_strides):
            self.assign(in_grad[i], req[i], self.in_grad_hist_list[i])

        gc.collect()


@mx.operator.register('fpn_rotated_roialign')
class FPNRotatedROIAlignProp(mx.operator.CustomOpProp):
    def __init__(self, feat_strides='(4,8,16,32)', pooled_height='7', pooled_width='7', output_dim='490'):
        super(FPNRotatedROIAlignProp, self).__init__(need_top_grad=True)
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.feat_strides = np.fromstring(feat_strides[1:-1], dtype=int, sep=',')
        self.output_dim = int(output_dim)

        self.num_strides = len(self.feat_strides)

    def list_arguments(self):
        args_list = []
        for i in range(self.num_strides):
            args_list.append('data_p{}'.format(2 + i))
        args_list.append('Rrois')
        return args_list

    def list_outputs(self):
        return ['rotated_pooled']

    def infer_shape(self, in_shape):
        # print 'num of Rrois:', in_shape[-1][0]
        output_feat_shape = [in_shape[-1][0], in_shape[0][1], self.pooled_height, self.pooled_width]
        return in_shape, [output_feat_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return FPNRotatedROIAlignOperator(self.feat_strides, self.pooled_height, self.pooled_width, self.output_dim)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [out_grad[0]]
