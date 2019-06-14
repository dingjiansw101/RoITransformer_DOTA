# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Haozhi Qi
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------
"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import numpy as np
import numpy.random as npr
import mxnet as mx

from utils.image import get_image, tensor_vstack
from bbox.bbox_transform import bbox_overlaps, bbox_transform, bbox_poly2hbb, dbbox_transform, dbbox_transform2
from bbox.bbox_transform import dbbox_transform2_warp, dbbox_transform2_inv_warp
from bbox.bbox_transform import dbboxtransform3_inv_warp, dbboxtransform3_warp, dbboxtransform3, dbboxtransform3_inv
from bbox.bbox_regression import expand_bbox_regression_targets, expand_bbox_regression_targets_base, expand_bbox_regression_targets_base_new
from bbox.bbox_transform import *
from dota_kit.poly_nms_gpu.poly_overlaps import poly_overlaps

import pdb

def get_rcnn_testbatch(roidb, cfg):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped'] + ['boxes']
    :return: data, label, im_info
    """
    # assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs
    im_info = [np.array([roidb[i]['im_info']], dtype=np.float32) for i in range(len(roidb))]

    im_rois = [roidb[i]['boxes'] for i in range(len(roidb))]
    rois = im_rois
    rois_array = [np.hstack((0 * np.ones((rois[i].shape[0], 1)), rois[i])) for i in range(len(rois))]

    data = [{'data': im_array[i],
             'rois': rois_array[i]} for i in range(len(roidb))]
    label = {}

    return data, label, im_info


def get_rcnn_batch(roidb, cfg):
    """
    return a dict of multiple images
    :param roidb: a list of dict, whose length controls batch size
    ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
    :return: data, label
    """
    num_images = len(roidb)
    imgs, roidb = get_image(roidb, cfg)
    im_array = tensor_vstack(imgs)

    assert cfg.TRAIN.BATCH_ROIS == -1 or cfg.TRAIN.BATCH_ROIS % cfg.TRAIN.BATCH_IMAGES == 0, \
        'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(cfg.TRAIN.BATCH_IMAGES, cfg.TRAIN.BATCH_ROIS)

    if cfg.TRAIN.BATCH_ROIS == -1:
        rois_per_image = np.sum([iroidb['boxes'].shape[0] for iroidb in roidb])
        fg_rois_per_image = rois_per_image
    else:
        rois_per_image = cfg.TRAIN.BATCH_ROIS / cfg.TRAIN.BATCH_IMAGES
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(int)

    rois_array = list()
    labels_array = list()
    bbox_targets_array = list()
    bbox_weights_array = list()

    for im_i in range(num_images):
        roi_rec = roidb[im_i]

        # infer num_classes from gt_overlaps
        num_classes = roi_rec['gt_overlaps'].shape[1]

        # label = class RoI has max overlap with
        rois = roi_rec['boxes']
        labels = roi_rec['max_classes']
        overlaps = roi_rec['max_overlaps']
        bbox_targets = roi_rec['bbox_targets']

        im_rois, labels, bbox_targets, bbox_weights = \
            sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                        labels, overlaps, bbox_targets)

        # project im_rois
        # do not round roi
        rois = im_rois
        batch_index = im_i * np.ones((rois.shape[0], 1))
        rois_array_this_image = np.hstack((batch_index, rois))
        rois_array.append(rois_array_this_image)

        # add labels
        labels_array.append(labels)
        bbox_targets_array.append(bbox_targets)
        bbox_weights_array.append(bbox_weights)

    rois_array = np.array(rois_array)
    labels_array = np.array(labels_array)
    bbox_targets_array = np.array(bbox_targets_array)
    bbox_weights_array = np.array(bbox_weights_array)

    data = {'data': im_array,
            'rois': rois_array}
    label = {'label': labels_array,
             'bbox_target': bbox_targets_array,
             'bbox_weight': bbox_weights_array}

    return data, label

def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    if labels is None:
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

def sample_rotbox_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                    labels=None, overlaps=None, dbbox_targets=None, gt_boxes=None):
    """

    :param rois: al_rois [n, 4]; e2e [n, 5] with batch_index
    :param fg_rois_per_image:
    :param rois_per_image:
    :param num_clases:
    :param cfg:
    :param labels:
    :param overlaps:
    :param dbbox_targets:
    :param gt_boxes: optional for e2e [n, 9] (x1, y1, ..., x4, y4, cls)
    :return:
    """
    if labels is None:
        # hgt_boxes = np.hstack((bbox_poly2hbb(gt_boxes[:, :-1]), gt_boxes[:, -1]))
        hgt_boxes = bbox_poly2hbb(gt_boxes)
        ## rois: (xmin, ymin, xmax, ymax)
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), hgt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = hgt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    # pdb.set_trace()
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]
    # pdb.set_trace()
    # load or compute bbox_target
    if dbbox_targets is not None:
        bbox_target_data = dbbox_targets[keep_indexes, :]
    else:
        targets = dbbox_transform2_warp(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))
    # pdb.set_trace()
    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_base(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

def sample_Rrois(Rrois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                    labels=None, overlaps=None, dbbox_targets=None, gt_boxes=None, device_id=0):
    """

    :param Rrois: all_rois [n, 5]; e2e [n, 6] with batch_index  (x_ctr, y_ctr, w, h, theta)
    :param fg_rois_per_image:
    :param rois_per_image:
    :param num_clases:
    :param cfg:
    :param labels:
    :param overlaps:
    :param dbbox_targets:
    :param gt_boxes: optional for e2e [n, 6] (x_ctr, y_ctr, w, h, theta, cls)
    :return:
    """
    if labels is None:
        ## rois: (xmin, ymin, xmax, ymax)
        # poly_overlaps = poly_overlaps_nms_wrapper(Rrois.context.device_id)
        overlaps = poly_overlaps(Rrois[:, 1:].astype(np.float32), gt_boxes[:, :5].astype(np.float32), device_id)
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 5]

    # pdb.set_trace()
    # foreground RoI with FG_THRESH overlap
    # fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    fg_indexes = np.where(overlaps >= cfg.TRAIN.RRoI_FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(Rrois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(Rrois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    # pdb.set_trace()
    labels[fg_rois_per_this_image:] = 0
    Rrois = Rrois[keep_indexes]
    # pdb.set_trace()
    # load or compute bbox_target
    if dbbox_targets is not None:
        bbox_target_data = dbbox_targets[keep_indexes, :]
    else:
        # targets = dbbox_transform2(Rrois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :5])
        targets = dbbox_transform2_best_match_warp(Rrois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :5])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            # targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
            #            / np.array(cfg.TRAIN.BBOX_STDS))
            targets = ((targets - np.array(cfg.TRAIN.RRoI_BBOX_STDS))
                       / np.array(cfg.TRAIN.RRoI_BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))
    # pdb.set_trace()
    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_base_new(bbox_target_data, num_classes, cfg.network.RRoI_CLASS_AGNOSTIC)

    return Rrois, labels, bbox_targets, bbox_weights

def sample_poly_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                    labels=None, overlaps=None, dbbox_targets=None, gt_boxes=None):
    """

    :param rois: al_rois [n, 4]; e2e [n, 5] with batch_index
    :param fg_rois_per_image:
    :param rois_per_image:
    :param num_clases:
    :param cfg:
    :param labels:
    :param overlaps:
    :param dbbox_targets:
    :param gt_boxes: optional for e2e [n, 9] (x1, y1, ..., x4, y4, cls)
    :return:
    """
    if labels is None:
        # hgt_boxes = np.hstack((bbox_poly2hbb(gt_boxes[:, :-1]), gt_boxes[:, -1]))
        hgt_boxes = bbox_poly2hbb(gt_boxes)
        ## rois: (xmin, ymin, xmax, ymax)
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), hgt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = hgt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if dbbox_targets is not None:
        bbox_target_data = dbbox_targets[keep_indexes, :]
    else:
        targets = dbbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))
    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_base(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

def sample_rotbox_rois_region_pred(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                    labels=None, overlaps=None, dbbox_targets=None, gt_boxes=None):
    """

    :param rois: al_rois [n, 4]; e2e [n, 5] with batch_index
    :param fg_rois_per_image:
    :param rois_per_image:
    :param num_clases:
    :param cfg:
    :param labels:
    :param overlaps:
    :param dbbox_targets:
    :param gt_boxes: optional for e2e [n, 9] (x1, y1, ..., x4, y4, cls)
    :return:
    """
    if labels is None:
        # hgt_boxes = np.hstack((bbox_poly2hbb(gt_boxes[:, :-1]), gt_boxes[:, -1]))
        hgt_boxes = bbox_poly2hbb(gt_boxes)
        ## rois: (xmin, ymin, xmax, ymax)
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), hgt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = hgt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    # pdb.set_trace()
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]
    # pdb.set_trace()
    # load or compute bbox_target
    if dbbox_targets is not None:
        bbox_target_data = dbbox_targets[keep_indexes, :]
    else:
        targets = dbbox_transform2_warp(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))
    # pdb.set_trace()
    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_base(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

def sample_rotbox_rois_nd(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                    labels=None, overlaps=None, dbbox_targets=None, gt_boxes=None):
    """

    :param rois: al_rois [n, 4]; e2e [n, 5] with batch_index
    :param fg_rois_per_image:
    :param rois_per_image:
    :param num_clases:
    :param cfg:
    :param labels:
    :param overlaps:
    :param dbbox_targets:
    :param gt_boxes: optional for e2e [n, 9] (x1, y1, ..., x4, y4, cls)
    :return:
    """
    if labels is None:
        # hgt_boxes = np.hstack((bbox_poly2hbb(gt_boxes[:, :-1]), gt_boxes[:, -1]))
        hgt_boxes = bbox_poly2hbb_nd(gt_boxes)
        ## rois: (xmin, ymin, xmax, ymax)
        # overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), hgt_boxes[:, :4].astype(np.float))
        overlaps = mx.nd.contrib.box_iou(rois[:, 1:].astype(np.float), hgt_boxes[:, :4].astype(np.float))

        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = hgt_boxes[gt_assignment, 4]

    overlaps = overlaps.asnumpy()

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # trans it to mx.nd
    keep_indexes = mx.nd.array(keep_indexes)
    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    # pdb.set_trace()
    if (fg_rois_per_this_image < labels.size):
        labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]
    # pdb.set_trace()
    # load or compute bbox_target
    if dbbox_targets is not None:
        bbox_target_data = dbbox_targets[keep_indexes, :]
    else:
        targets = dbbox_transform2_warp_nd(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - mx.nd.array(cfg.TRAIN.BBOX_MEANS))
                       / mx.nd.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = mx.nd.concat(labels.expand_dims(1), targets, dim=1)
        # bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_base(bbox_target_data.asnumpy(), num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

def sample_xyhs_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                     labels=None, overlaps=None, dbbox_targets=None, gt_boxes=None):
    """

    :param rois: al_rois [n, 4]; e2e [n, 5] with batch_index
    :param fg_rois_per_image:
    :param rois_per_image:
    :param num_clases:
    :param cfg:
    :param labels:
    :param overlaps:
    :param dbbox_targets:
    :param gt_boxes: optional for e2e [n, 9] (x1, y1, ..., x4, y4, cls)
    :return:
    """
    if labels is None:
        # hgt_boxes = np.hstack((bbox_poly2hbb(gt_boxes[:, :-1]), gt_boxes[:, -1]))
        hgt_boxes = bbox_poly2hbb(gt_boxes)
        ## rois: (xmin, ymin, xmax, ymax)
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), hgt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = hgt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if dbbox_targets is not None:
        bbox_target_data = dbbox_targets[keep_indexes, :]
    else:
        # targets = dbbox_transform2_warp(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        targets = dbboxtransform3_warp(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))
    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_base(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights

def sample_xyhs_rois_nd(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                     labels=None, overlaps=None, dbbox_targets=None, gt_boxes=None):
    """
    this is an mxnet version
    :param rois: al_rois [n, 4]; e2e [n, 5] with batch_index
    :param fg_rois_per_image:
    :param rois_per_image:
    :param num_clases:
    :param cfg:
    :param labels:
    :param overlaps:
    :param dbbox_targets:
    :param gt_boxes: optional for e2e [n, 9] (x1, y1, ..., x4, y4, cls)
    :return:
    """
    if labels is None:
        # hgt_boxes = np.hstack((bbox_poly2hbb(gt_boxes[:, :-1]), gt_boxes[:, -1]))
        hgt_boxes = bbox_poly2hbb_nd(gt_boxes)
        ## rois: (xmin, ymin, xmax, ymax)
        overlaps = mx.nd.contrib.box_iou(rois[:, 1:].astype('float32'), hgt_boxes[:, :4].astype('float32'))
        # overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), hgt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = hgt_boxes[gt_assignment, 4]

    # tmp trans it to numpy
    overlaps = overlaps.asnumpy()
    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if dbbox_targets is not None:
        bbox_target_data = dbbox_targets[keep_indexes, :]
    else:
        # targets = dbbox_transform2_warp(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        targets = dbboxtransform3_warp(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :8])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))
    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets_base(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights



