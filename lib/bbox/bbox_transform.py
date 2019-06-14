# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# py-faster-rcnn
# Copyright (c) 2016 by Contributors
# Licence under The MIT License
# py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import numpy as np
from bbox import bbox_overlaps_cython
import math
import copy
import mxnet as mx
import pdb

def bbox_overlaps(boxes, query_boxes):
    return bbox_overlaps_cython(boxes, query_boxes)

def bbox_poly2hbb(boxes):
    """
    with label
    :param boxes: (x1, y1, ... x4, y4, cls) [n, 9]
    :return: hbb: (xmin, ymin, xmax, ymax, cls) [n, 5]
    """
    n = boxes.shape[0]
    hbbs = np.zeros((n, 4))

    xs = np.reshape(boxes[:, : -1], (n, 4, 2))[:, :, 0]
    ys = np.reshape(boxes[:, : -1], (n, 4, 2))[:, :, 1]
    # pdb.set_trace()
    hbbs[:, 0] = np.min(xs, axis=1)
    hbbs[:, 1] = np.min(ys, axis=1)
    hbbs[:, 2] = np.max(xs, axis=1)
    hbbs[:, 3] = np.max(ys, axis=1)
    hbbs = np.hstack((hbbs, boxes[:, -1, np.newaxis]))
    return hbbs
def bbox_poly2hbb_nd(boxes):
    """
    with label. this is a mxnet version
    :param boxes: (x1, y1, ..., x4, y4, cls) [n, 9]
    :return: hbb: (xmin, ymin, xmax, ymax, cls) [n, 5]
    """
    n = boxes.shape[0]
    hbbs = mx.nd.zeros((n, 4))

    xs = boxes[:, : -1].reshape((n, 4, 2))[:, :, 0]
    ys = boxes[:, : -1].reshape((n, 4, 2))[:, :, 1]

    hbbs[:, 0] = mx.nd.min(xs, axis=1)
    hbbs[:, 1] = mx.nd.min(ys, axis=1)
    hbbs[:, 2] = mx.nd.max(xs, axis=1)
    hbbs[:, 3] = mx.nd.max(ys, axis=1)
    hbbs = mx.nd.concat(hbbs, mx.nd.expand_dims(boxes[:, -1], 1), dim=1)

    return hbbs
def box2poly(boxes):
    """
    :param boxes: (x, y, w, h)  [n, 4]
    :return: (x1, y1, ... x4, y4) [n, 8]
    """
    xs = boxes[:, 0]
    ys = boxes[:, 1]
    ws = boxes[:, 2]
    hs = boxes[:, 3]
    n = len(xs)
    polys = np.zeros((n, 8))
    polys[:, 0] = xs - ws/2.0
    polys[:, 1] = ys - hs/2.0
    polys[:, 2] = xs + ws/2.0
    polys[:, 3] = ys - hs/2.0
    polys[:, 4] = xs + ws/2.0
    polys[:, 5] = ys + hs/2.0
    polys[:, 6] = xs - ws/2.0
    polys[:, 7] = ys + hs/2.0

    return polys

def xy2wh(boxes):
    """
    :param boxes: (xmin, ymin, xmax, ymax) (n,4)
    :return: out_boxes: (x_ctr, y_ctr, w, h) (n, 4)
    """
    num_boxes = boxes.shape[0]

    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = boxes[:, 1] + 0.5 * (ex_heights - 1.0)

    return np.concatenate((ex_ctr_x[:, np.newaxis], ex_ctr_y[:, np.newaxis], ex_widths[:, np.newaxis], ex_heights[:, np.newaxis]), axis=1)

def xy2wh_nd(boxes):
    """

    :param boxes: (xmin, ymin, xmax, ymax) (n, 4)
    :return: out_boxes: (x_ctr, y_ctr, w, h) (n, 4)
    """
    num_boxes = boxes.shape[0]

    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = boxes[:, 1] + 0.5 * (ex_heights - 1.0)

    return mx.nd.concat(ex_ctr_x.expand_dims(1),
                         ex_ctr_y.expand_dims(1),
                         ex_widths.expand_dims(1),
                         ex_heights.expand_dims(1), dim=1)

def wh2xy(boxes):
    """

    :param boxes: (x_ctr, y_ctr, w, h) (n, 4)
    :return: out_boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    num_boxes = boxes.shape[0]

    xmin = boxes[:, 0] - (boxes[:, 2] - 1) / 2.0
    ymin = boxes[:, 1] - (boxes[:, 3] - 1) / 2.0
    xmax = boxes[:, 0] + (boxes[:, 2] - 1) / 2.0
    ymax = boxes[:, 1] + (boxes[:, 3] - 1) / 2.0

    return np.concatenate((xmin[:, np.newaxis], ymin[:, np.newaxis], xmax[:, np.newaxis], ymax[:, np.newaxis]), axis=1)
def poly2bbox(polys):
    """
    without label
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = np.reshape(polys, (n, 4, 2))[:, :, 0]
    ys = np.reshape(polys, (n, 4, 2))[:, :, 1]

    xmin = np.min(xs, axis=1)
    ymin = np.min(ys, axis=1)
    xmax = np.max(xs, axis=1)
    ymax = np.max(ys, axis=1)

    xmin = xmin[:, np.newaxis]
    ymin = ymin[:, np.newaxis]
    xmax = xmax[:, np.newaxis]
    ymax = ymax[:, np.newaxis]

    return np.concatenate((xmin, ymin, xmax, ymax), 1)

def poly2bbox_nd(polys):
    """
    this is a mx.nd version
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = polys.reshape(n, 4, 2)[:, :, 0]
    ys = polys.reshape(n, 4, 2)[:, :, 1]

    xmin = mx.nd.min(xs, axis=1)
    ymin = mx.nd.min(ys, axis=1)
    xmax = mx.nd.max(xs, axis=1)
    ymax = mx.nd.max(ys, axis=1)

    xmin = xmin.expand_dims(1)
    ymin = ymin.expand_dims(1)
    xmax = xmax.expand_dims(1)
    ymax = ymax.expand_dims(1)

    return mx.nd.concat(xmin, ymin, xmax, ymax, dim=1)
#
# def dbbox_transform(ex_rois, gt_rois):
#     """
#     :param ex_rois: predicted rois from rpn (x, y, w, h)
#             shape [n, 4]
#     :param gt_rois: ground truth rois (x1, y1, x2, y2, x3, y3, x4, y4)
#             shape [n, 8]
#     :return: dbbox target [n, 8]
#     """
#     roi_polys = box2poly(ex_rois)
#     ws = ex_rois[:, 2]
#     hs = ex_rois[:, 3]
#     n = len(ws)
#     targets = np.zeros((n, 8))
#     for i in range(8):
#         if i%2 == 0:
#             # dx
#             targets[:, i] = (gt_rois[:, i] - roi_polys[:, i]) / ws
#         else:
#             # dy
#             targets[:, i] = (gt_rois[:, i] - roi_polys[:, i]) / hs
#     return targets

# def rotbox_norm(rotboxes):
#     """
#     if the rot
#     :param rotboxes:
#     :return:
#     """


def dbbox_transform(ex_rois, gt_rois):
    """
    :param ex_rois: predicted rois from rpn (xmin, ymin, xmax, ymax)
            shape [n, 4]
    :param gt_rois: ground truth rois (x1, y1, x2, y2, x3, y3, x4, y4)
            shape [n, 8]
    :return: dbbox target [n, 8]
    """
    ws = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    hs = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    xmin, ymin, xmax, ymax = ex_rois[:, 0], ex_rois[:, 1], ex_rois[:, 2], ex_rois[:, 3]
    roi_polys = np.concatenate((xmin[:, np.newaxis],
                                ymin[:, np.newaxis],
                                xmax[:, np.newaxis],
                                ymin[:, np.newaxis],
                                xmax[:, np.newaxis],
                                ymax[:, np.newaxis],
                                xmin[:, np.newaxis],
                                ymax[:, np.newaxis]), 1)

    n = len(ws)
    targets = np.zeros((n, 8))
    for i in range(8):
        if i%2 == 0:
            # dx
            targets[:, i] = (gt_rois[:, i] - roi_polys[:, i]) / ws
        else:
            # dy
            targets[:, i] = (gt_rois[:, i] - roi_polys[:, i]) / hs
    return targets

def dbbox_pred(boxes, box_deltas):
    """
    Transform the set of calss-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: rois (xmin, ymin, xmax, ymax)  [n, 4]
    :param box_deltas: (dx1, dy1, ..., dx4, dy4)    [n, 8]
    :return: box_pred: (x1, y1, ..., x4, y4)    [n, 8]
    x_pred = dx * w + x
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    # ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    # ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[: , 2], boxes[:, 3]
    rois = np.concatenate((xmin[:, np.newaxis], ymin[:, np.newaxis],
                           xmax[:, np.newaxis], ymin[:, np.newaxis],
                           xmax[:, np.newaxis], ymax[:, np.newaxis],
                           xmin[:, np.newaxis], ymax[:, np.newaxis]), axis=1)
    box_pred = np.zeros(box_deltas.shape)
    for i in range(8):
        if i %2 == 0:
            # for x
            # pdb.set_trace()
            box_pred[:, i::8] = box_deltas[:, i::8] * widths[:, np.newaxis] + rois[:, i, np.newaxis]
        else:
            # for y
            box_pred[:, i::8] = box_deltas[:, i::8] * heights[:, np.newaxis] + rois[:, i, np.newaxis]
    return box_pred

def polygonToRotRectangle_batch(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]
    """
    # print('bbox: ', bbox)
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)


    xmin = np.min(normalized[:, 0, :], axis=1)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[:, 0, :], axis=1)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    angle = angle[:, np.newaxis] % ( 2 * np.pi)

    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes

# def polygonToRectangle_nd(bbox):
#     """
#     :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
#             shape [num_boxes, 8]
#     :return: Rotated Rectangle in format [cx, cy, w, h, theta]
#             shape [num_rot_recs, 5]
#     """
#
#     bbox_numpy = mx.nd.array(bbox, dtype='float32')
#


def RotBox2Polys(dboxes):
    """
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    cs = np.cos(dboxes[:, 4])
    ss = np.sin(dboxes[:, 4])
    w = dboxes[:, 2] - 1
    h = dboxes[:, 3] - 1

    ## change the order to be the initial definition
    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    x1 = x1[:, np.newaxis]
    y1 = y1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    y2 = y2[:, np.newaxis]
    x3 = x3[:, np.newaxis]
    y3 = y3[:, np.newaxis]
    x4 = x4[:, np.newaxis]
    y4 = y4[:, np.newaxis]



    polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)
    return polys

def xyhs2polys(xyhs):
    """

    :param xyhs:  (x1, y1, x2, y2, h) (x1, y1) and (x2, y2) are the first and second point, the h is the height of boudning box
            shape: (num_rois, 5)
    :return: polys: (x1, y1, x2, y2, x3, y3, x4, y4)
            shape: (num_rois, 8)
    """
    x1 = xyhs[:, 0]
    y1 = xyhs[:, 1]
    x2 = xyhs[:, 2]
    y2 = xyhs[:, 3]
    h = xyhs[:, 4]

    A = -(y2 - y1)
    B = (x2 - x1)
    x3 = x2 + A/(np.sqrt(A * A + B * B)) * h
    y3 = y2 + B/(np.sqrt(A * A + B * B)) * h
    x4 = x1 + A/(np.sqrt(A * A + B * B)) * h
    y4 = y1 + B/(np.sqrt(A * A + B * B)) * h

    return np.concatenate((x1[:, np.newaxis],
                           y1[:, np.newaxis],
                           x2[:, np.newaxis],
                           y2[:, np.newaxis],
                           x3[:, np.newaxis],
                           y3[:, np.newaxis],
                           x4[:, np.newaxis],
                           y4[:, np.newaxis]), axis=1)

def xyhs2polys_muli_class(xyhs):
    """
    :param xyhs: (x1, y1, x2, y2, h)
            (numboxes, 5 * num_classes)
    :return: quadrangles:
            (numboxes, 8 * num_classes)
    """
    num_boxes = xyhs.shape[0]
    numclasses = int(xyhs.shape[1]/5)
    quadrangles = np.zeros((num_boxes, 8 * numclasses))
    x1 = xyhs[:, 0::5]
    y1 = xyhs[:, 1::5]
    x2 = xyhs[:, 2::5]
    y2 = xyhs[:, 3::5]
    h = xyhs[:, 4::5]

    A = -(y2 - y1)
    B = (x2 - x1)
    x3 = x2 + A/(np.sqrt(A * A + B * B)) * h
    y3 = y2 + B/(np.sqrt(A * A + B * B)) * h
    x4 = x1 + A/(np.sqrt(A * A + B * B)) * h
    y4 = y1 + B/(np.sqrt(A * A + B * B)) * h

    quadrangles[:, 0::8] = x1
    quadrangles[:, 1::8] = y1
    quadrangles[:, 2::8] = x2
    quadrangles[:, 3::8] = y2
    quadrangles[:, 4::8] = x3
    quadrangles[:, 5::8] = y3
    quadrangles[:, 6::8] = x4
    quadrangles[:, 7::8] = y4

    return quadrangles

def polys2xyhs(polys):
    """
    Transform polys to xyhs, note! it is reversible to function xyhs2polys if only if the polys are rectangle.
    :param polys:
    :return:
    """
    rotboxes = polygonToRotRectangle_batch(polys)
    polys = RotBox2Polys(rotboxes)
    x1 = polys[:, 0]
    y1 = polys[:, 1]
    x2 = polys[:, 2]
    y2 = polys[:, 3]
    h = np.sqrt((polys[:, 2] - polys[:, 4])**2 + (polys[:, 3] - polys[:, 5])**2)

    return np.concatenate((x1[:, np.newaxis],
                           y1[:, np.newaxis],
                           x2[:, np.newaxis],
                           y2[:, np.newaxis],
                           h[:, np.newaxis]), axis=1)

def polys2xyhs_nd(polys):
    """

    :param polys:
    :return:
    """
    x1 = polys[:, 0]
    y1 = polys[:, 1]
    x2 = polys[:, 2]
    y2 = polys[:, 3]
    h = np.sqrt((polys[:, 2] - polys[:, 4])**2 + (polys[:, 3] - polys[:, 5])**2)

    return mx.nd.concat((mx.nd.expand_dims(x1, 1),
                         mx.nd.expand_dims(y1, 1),
                         mx.nd.expand_dims(x2, 1),
                         mx.nd.expand_dims(y2, 1),
                         mx.nd.expand_dims(h, 1)), 1)

def dbboxtransform3_warp(ex_rois, gts):
    """

    :param ex_rois: (xmin, ymin, xmax, ymax)
            shape: (num_rois, 5)
    :param gts: (x1, y1, ..., x4, y4)
            shape: (num_rois, 8)
    :return: targets: (dx1, dy1, dx2, dy2, dh)
    """
    boxes_x1s = ex_rois[:, 0]
    boxes_y1s = ex_rois[:, 1]
    boxes_x2s = ex_rois[:, 2]
    boxes_y2s = ex_rois[:, 1]
    hs = ex_rois[:, 3] - ex_rois[:, 1]

    boxes_xyhs = np.concatenate((boxes_x1s[:, np.newaxis],
                                 boxes_y1s[:, np.newaxis],
                                 boxes_x2s[:, np.newaxis],
                                 boxes_y2s[:, np.newaxis],
                                 hs[:, np.newaxis]), axis=1)

    gts_xyhs = polys2xyhs(gts)

    targets = dbboxtransform3(boxes_xyhs, gts_xyhs)

    return targets

def dbbox_transform3_warp_nd(ex_rois, gts):
    """
    this is a mxnet version
    :param ex_rois:
    :param gts:
    :return:
    """
    boxes_x1s = ex_rois[:, 0]
    boxes_y1s = ex_rois[:, 1]
    boxes_x2s = ex_rois[:, 2]
    boxes_y2s = ex_rois[:, 1]
    hs = ex_rois[:, 3] - ex_rois[:, 1]

    # boxes_xyhs = np.concatenate((boxes_x1s[:, np.newaxis],
    #                              boxes_y1s[:, np.newaxis],
    #                              boxes_x2s[:, np.newaxis],
    #                              boxes_y2s[:, np.newaxis],
    #                              hs[:, np.newaxis]), axis=1)

    boxes_xyhs = mx.nd.concat((mx.nd.expand_dims(boxes_x1s, 1),
                               mx.nd.expand_dims(boxes_y1s, 1),
                               mx.nd.expand_dims(boxes_x2s, 1),
                               mx.nd.expand_dims(boxes_y2s, 1),
                               mx.nd.expand_dims(hs, 1)), dim=1)

    gts_xyhs = polys2xyhs_nd(gts)

    targets = dbboxtransform3_nd(boxes_xyhs, gts_xyhs)

    return targets

def dbboxtransform3_inv_warp(ex_rois, deltas):
    """

    :param ex_rois: (xmin, ymin, xmax, ymax)
            shape: (num_rois, 4)
    :param deltas: (dx1, dy1, dx2, dy2, dh)
            shape: (num_rois, 5 * num_classes)
    :return: pred_boxes: (x1, y1, x2, y2, h)
    """
    boxes_x1s = ex_rois[:, 0]
    boxes_y1s = ex_rois[:, 1]
    boxes_x2s = ex_rois[:, 2]
    boxes_y2s = ex_rois[:, 1]
    hs = ex_rois[:, 3] - ex_rois[:, 1]

    boxes_xyhs = np.concatenate((boxes_x1s[:, np.newaxis],
                                 boxes_y1s[:, np.newaxis],
                                 boxes_x2s[:, np.newaxis],
                                 boxes_y2s[:, np.newaxis],
                                 hs[:, np.newaxis]), axis=1)

    pred_boxes = dbboxtransform3_inv(boxes_xyhs, deltas)

    return pred_boxes

def rotation_translation_trans(boxes, thetas, translations):
    """
    apply rotation and translation transform on boxes
    :param boxes: (x1, y1, x2, y2, h)
            shape: (n, 5)
    :return: output boxes: (x1, y1, x2, y2, h)
            shape: (n, 5)
    """

    boxes[:, 0] = boxes[:, 0] - translations[:, 0]
    boxes[:, 1] = boxes[:, 1] - translations[:, 1]
    xs, ys = copy.deepcopy(boxes[:, 0]), copy.deepcopy(boxes[:, 1])
    boxes[:, 0] = np.cos(thetas) * xs - np.sin(thetas) * ys
    boxes[:, 1] = np.sin(thetas) * xs + np.cos(thetas) * ys

    boxes[:, 2] = boxes[:, 2] - translations[:, 0]
    boxes[:, 3] = boxes[:, 3] - translations[:, 1]
    xs2, ys2 = copy.deepcopy(boxes[:, 2]), copy.deepcopy(boxes[:, 3])
    boxes[:, 2] = np.cos(thetas) * xs2 - np.sin(thetas) * ys2
    boxes[:, 3] = np.sin(thetas) * xs2 + np.cos(thetas) * ys2

    return boxes
def rotation_translation_trans_multi_class(boxes, thetas, translations):
    """
    apply rotation and translation transform on boxes
    :param boxes: (x1, y1)
    :param thetas:
    :param translations:
    :return:
    """
    boxes[:, 0::5] = boxes[:, 0::5] - translations[:, 0][:, np.newaxis]
    boxes[:, 1::5] = boxes[:, 1::5] - translations[:, 1][:, np.newaxis]
    xs, ys = boxes[:, 0::5], boxes[:, 1::5]
    boxes[:, 0::5] = np.cos(thetas)[:, np.newaxis] * xs - np.sin(thetas)[:, np.newaxis] * ys
    boxes[:, 1::5] = np.sin(thetas)[:, np.newaxis] * xs + np.cos(thetas)[:, np.newaxis] * ys

    boxes[:, 2::5] = boxes[:, 2::5] - translations[:, 0][:, np.newaxis]
    boxes[:, 3::5] = boxes[:, 3::5] - translations[:, 1][:, np.newaxis]
    xs2, ys2 = boxes[:, 2::5], boxes[:, 3::5]
    boxes[:, 2::5] = np.cos(thetas)[:, np.newaxis] * xs2 - np.sin(thetas)[:, np.newaxis] * ys2
    boxes[:, 3::5] = np.sin(thetas)[:, np.newaxis] * xs2 + np.cos(thetas)[:, np.newaxis] * ys2

    return boxes

def dbboxtransform3(boxes, gts):
    """
    boxes are rotated RoIs,
    :param boxes: (x1, y1, x2, y2, h)
    :param gts: (x1, y1, x2, y2, h)
    :return: targets: (dx1, dy1, dx2, dy2, dh)
    """
    # pdb.set_trace()
    thetas = -np.arctan2((boxes[:, 3] - boxes[:, 1]), (boxes[:, 2] - boxes[:, 0]))
    ext_widhts = np.sqrt((boxes[:, 0] - boxes[:, 2])**2 + (boxes[:, 1] - boxes[:, 3])**2)
    hs = boxes[:, 4]
    # transform the coords of gts to the coordinates of boxes
    # combination of rotation and translation
    # pdb.set_trace()
    transformed_gts = copy.deepcopy(gts)

    transformed_gts = rotation_translation_trans(transformed_gts, thetas, boxes[:, 0:2])

    transformed_boxes = copy.deepcopy(boxes)
    transformed_boxes[:, 0] = 0
    transformed_boxes[:, 1] = 0
    transformed_boxes[:, 2] = ext_widhts
    transformed_boxes[:, 3] = 0
    transformed_boxes[:, 4] = hs

    dx1 = (transformed_gts[:, 0] - transformed_boxes[:, 0]) / ext_widhts
    dy1 = (transformed_gts[:, 1] - transformed_boxes[:, 1]) / hs
    dx2 = (transformed_gts[:, 2] - transformed_boxes[:, 2]) / ext_widhts
    dy2 = (transformed_gts[:, 3] - transformed_boxes[:, 3]) / hs
    dh = np.log(transformed_gts[:, 4] / hs)

    return np.concatenate((dx1[:, np.newaxis], dy1[:, np.newaxis], dx2[:, np.newaxis], dy2[:, np.newaxis], dh[:, np.newaxis]), axis=1)

def dbboxtransform3_nd(boxes, gts):
    """
    this is a mxnet version
    boxes are rotated RoIs,
    :param boxes: (x1, y1, x2, y2, h)
    :param gts: (x1, y1, x2, y2, h)
    :return: targets: (dx1, dy1, dx2, dy2, dh)
    """
    # thetas = np.

def dbboxtransform3_inv(boxes, targets):
    """

    :param boxes: (x1, y1, x2, y2, h)
            shape: (num_rois, 5)
    :param targets: (dx1, dy1, dx2, dy2, dh)
            shape: (num_rois, 5 * num_classes)
    :return: decoded predicts (x1, y1, x2, y2, h)
            shape: (num_rois, 5 * num_classes)
    """
    # TODO: handle corner cases
    thetas = -np.arctan2((boxes[:, 3] - boxes[:, 1]), (boxes[:, 2] - boxes[:, 0]))
    ext_widhts = np.sqrt((boxes[:, 0] - boxes[:, 2])**2 + (boxes[:, 1] - boxes[:, 3])**2)
    hs = boxes[:, 4]
    # calculate the coords in the coordinates bind to the boxes
    transformed_boxes = copy.deepcopy(boxes)
    transformed_boxes[:, 0] = 0
    transformed_boxes[:, 1] = 0
    transformed_boxes[:, 2] = ext_widhts
    transformed_boxes[:, 3] = 0
    transformed_boxes[:, 4] = hs

    transformed_gts = copy.deepcopy(targets)
    transformed_gts[:, 0::5] = targets[:, 0::5] * ext_widhts[:, np.newaxis] + transformed_boxes[:, 0][:, np.newaxis]
    transformed_gts[:, 1::5] = targets[:, 1::5] * hs[:, np.newaxis] + transformed_boxes[:, 1][:, np.newaxis]
    transformed_gts[:, 2::5] = targets[:, 2::5] * ext_widhts[:, np.newaxis] + transformed_boxes[:, 2][:, np.newaxis]
    transformed_gts[:, 3::5] = targets[:, 3::5] * hs[:, np.newaxis] + transformed_boxes[:, 3][:, np.newaxis]
    transformed_gts[:, 4::5] = np.exp(targets[:, 4::5]) * hs[:, np.newaxis]

    # transform from the coordinates bind to the boxes to the coordinates bind to the images
    pred_boxes = rotation_translation_trans_multi_class(transformed_gts, -thetas, -boxes[:, 0:2])
    #
    #
    # transformed_gts[:, 0::5] = transformed_gts[:, 0::5] + boxes[:, 0][:, np.newaxis]
    # transformed_gts[:, 1::5] = transformed_gts[:, 1::5] + boxes[:, 1][:, np.newaxis]
    # xs, ys = transformed_gts[:, 0::5], transformed_gts[:, 1::5]
    # transformed_gts[:, 0::5] = np.cos(-thetas[:, np.newaxis]) * xs - np.sin(-thetas[:, np.newaxis]) * ys
    # transformed_gts[:, 1::5] = np.sin(-thetas[:, np.newaxis]) * xs + np.cos(-thetas[:, np.newaxis]) * ys
    #
    # transformed_gts[:, 2::5] = transformed_gts[:, 2::5] + boxes[:, 0][:, np.newaxis]
    # transformed_gts[:, 3::5] = transformed_gts[:, 3::5] + boxes[:, 1][:, np.newaxis]
    # xs2, ys2 = transformed_gts[:, 2::5], transformed_gts[:, 3::5]
    # transformed_gts[:, 2::5] = np.cos(-thetas[:, np.newaxis]) * xs2 - np.sin(-thetas[:, np.newaxis]) * ys2
    # transformed_gts[:, 3::5] = np.sin(-thetas[:, np.newaxis]) * xs2 + np.cos(-thetas[:, np.newaxis]) * ys2

    return pred_boxes

def RotBox2Polys_multi_class(dboxes):
    """
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5 * num_classes)
    :return: quadranlges:
        (numboxes, 8 * num_classes)
    """
    num_boxes = dboxes.shape[0]
    numclasses = int(dboxes.shape[1]/5)
    quadrangles = np.zeros((num_boxes, 8 * numclasses))
    cs = np.cos(dboxes[:, 4::5])
    ss = np.sin(dboxes[:, 4::5])
    w = dboxes[:, 2::5] - 1
    h = dboxes[:, 3::5] - 1

    ## change the order to be the initial definition
    x_ctr = dboxes[:, 0::5]
    y_ctr = dboxes[:, 1::5]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    quadrangles[:, 0::8] = x1
    quadrangles[:, 1::8] = y1
    quadrangles[:, 2::8] = x2
    quadrangles[:, 3::8] = y2
    quadrangles[:, 4::8] = x3
    quadrangles[:, 5::8] = y3
    quadrangles[:, 6::8] = x4
    quadrangles[:, 7::8] = y4

    return quadrangles


# def encodebox(ex_rois, gt_rois):
#     """
#     first pred (dx, dy, dw, dh) then predict the middle point of each side
#     :param ex_rois: (xmin, ymin, xmax, ymax)
#             shape: (n, 4)
#     :param gt_rois: (x1, y1, x2, y2, ... x4, y4)
#             shape: (n, 8)
#     first transfer the gt_rois to (x, y, w, h, theta)
#     :return: (dx, dy, dw, dh, )
#     """
#     gt_rois_t = RotBox2Polys(polygonToRotRectangle_batch(gt_rois)) # (x1, y1, x2, y2, ..., x4, y4)
#     ## find left top point
#     xs = gt_rois_t.reshape(-1, 4, 2)[:, :, 0]
#     ys = gt_rois_t.reshape(-1, 4, 2)[:, :, 1]
#
#     top_index = np.argmin(ys, axis=1)
#
#     hbbs = poly2bbox(gt_rois_t)
#
#     targets1 = nonlinear_transform(ex_rois, hbbs)
#     top = xs[top_index] -


def dbbox_transform2_warp(ex_rois, gt_rois):
    """
    used to change the interface
    :param ex_rois: (xmin, ymin, xmax, ymax)
            shape (n, 4)
    :param gt_rois: (x1, y1, ... x4, y4)
            shape (n, 8)
    :return: encoded targets: shape (n, 5)
    """
    num_rois = ex_rois.shape[0]
    # TODO: carefully set it
    initial_angles = -np.ones((num_rois, 1)) * np.pi / 2.
    ex_rois = xy2wh(ex_rois)
    ex_rois = np.hstack((ex_rois, initial_angles))
    # pdb.set_trace()
    gt_rotboxes = polygonToRotRectangle_batch(gt_rois)
    targets = dbbox_transform2(ex_rois, gt_rotboxes)

    return targets

def dbbox_transform2_warp_nd(ex_rois, gt_rois):
    """
    usde to change the interface, this is a mx.nd version
    :param ex_rois: (xmin, ymin, xmax, ymax)
            shape (n, 4)
    :param gt_rois: (x1, y1, ..., x4, y4)
            shape (n, 8)
    :return: encoded targets: shape (n,5)
    """
    num_rois = ex_rois.shape[0]
    # TODO: carefully set it
    initial_angles = -mx.nd.ones((num_rois, 1)) * np.pi/2
    ex_rois = xy2wh_nd(ex_rois)
    # pdb.set_trace()
    ex_rois = mx.nd.concat(ex_rois, initial_angles, dim=1)

    gt_rotboxes = mx.nd.array(polygonToRotRectangle_batch(gt_rois.asnumpy()))
    targets = dbbox_transform2_nd(ex_rois, gt_rotboxes)

    return targets

def dbbox_transform2_inv_warp(ex_rois, deltas):
    """
    used to change the interface
    :param ex_rois: (xmin, ymin, xmax, ymax)
            shape (n, 4)
    :param deltas: (dx, dy, dw, dh, dtheta)
            shape (n, 5 * num_classes)
    :return:  decoded rotboxs: shape (n, 5 * num_classes)
    """
    num_rois = ex_rois.shape[0]
    initial_angles = -np.ones((num_rois, 1)) * np.pi / 2.
    ex_rois = xy2wh(ex_rois)
    ex_rois = np.hstack((ex_rois, initial_angles))
    pred_rotboxes = dbbox_transform2_inv(ex_rois, deltas)

    return pred_rotboxes

# def rotboxdecodewarp(ex_drois, deltas):
#     """
#
#     :param ex_drois: (xmin, ymin, xmax, ymax)
#             shape (n, 4)
#     :param deltas: (dx, dy, dw, dh, dtheta)
#             shape (n, 5 * num_classes)
#     :return: decoded polys: shape (n, 8 * num_classes)
#     """
#     pred_rotboxes = dbbox_transform2_inv_warp(ex_drois, deltas)


def dbbox_transform2(ex_drois, gt_rois):
    """
    :param poly_rois: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :param gt_rois: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :return: encoded targets: shape (n, 5)
    """
    # pdb.set_trace()
    gt_widths = gt_rois[:, 2]
    gt_heights = gt_rois[:, 3]
    gt_angle = gt_rois[:, 4]

    ex_widths = ex_drois[:, 2]
    ex_heights = ex_drois[:, 3]
    ex_angle = ex_drois[:, 4]

    coord = gt_rois[:, 0: 2] - ex_drois[:, 0:2]
    targets_dx = (np.cos(ex_drois[:, 4]) * coord[:, 0] + np.sin(ex_drois[:, 4]) * coord[:, 1]) / ex_widths
    targets_dy = (-np.sin(ex_drois[:, 4]) * coord[:, 0] + np.cos(ex_drois[:, 4]) * coord[:, 1]) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    targets_dangle = (gt_angle - ex_angle) % (2 * np.pi) / (2 * np.pi)
    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_dangle), 1)

    return targets

def dbbox_transform2_new(ex_drois, gt_rois):
    """
    :param poly_rois: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :param gt_rois: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :return: encoded targets: shape (n, 5)
    """
    # pdb.set_trace()
    gt_widths = gt_rois[:, 2]
    gt_heights = gt_rois[:, 3]
    gt_angle = gt_rois[:, 4]

    ex_widths = ex_drois[:, 2]
    ex_heights = ex_drois[:, 3]
    ex_angle = ex_drois[:, 4]

    coord = gt_rois[:, 0: 2] - ex_drois[:, 0:2]
    targets_dx = (np.cos(ex_drois[:, 4]) * coord[:, 0] + np.sin(ex_drois[:, 4]) * coord[:, 1]) / ex_widths
    targets_dy = (-np.sin(ex_drois[:, 4]) * coord[:, 0] + np.cos(ex_drois[:, 4]) * coord[:, 1]) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    targets_dangle = (gt_angle - ex_angle)
    dist = targets_dangle % (2 * np.pi)
    dist = np.minimum(dist, np.pi * 2 - dist)
    try:
        assert np.all(dist <= (np.pi/2. + 0.001))
    except:
        pdb.set_trace()
    # check clockwise or anti-clockwise, if sin(dtheta) < 0, clockwise
    mask = np.sin(targets_dangle) < 0
    dist[mask] = -dist[mask]
    # TODO: change the norm value
    dist = dist / (np.pi / 2.)
    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh, dist), 1)

    return targets

def dbbox_transform2_inv_new(ex_drois, deltas, norm_angle):
    """
    inspired from light-head rcnn, different classes share the same bbox regression
    :param ex_rois: (x, y, w, h, theta) shape (n, 5)
    :param deltas: (dx, dy, dw, dh, dtheta, dx ...) (n, 5 * numclasses)
    :return:
    """
    widths = ex_drois[:, 2]
    heights = ex_drois[:, 3]
    angles = ex_drois[:, 4]
    ctr_x = ex_drois[:, 0]
    ctr_y = ex_drois[:, 1]

    dx = deltas[:, 0::5]
    dy = deltas[:, 1::5]
    dw = deltas[:, 2::5]
    dh = deltas[:, 3::5]

    # clip dw
    # pdb.set_trace()
    # fuck, I write maximum at first
    # dw = np.minimum(dw, 4)
    # dh = np.minimum(dh, 4)

    dangle = deltas[:, 4::5]
    # pdb.set_trace()
    pred_ctr_x = dx * widths[:, np.newaxis] * np.cos(angles[:, np.newaxis]) \
                 - dy * heights[:, np.newaxis] * np.sin(angles[:, np.newaxis]) + ctr_x[:, np.newaxis]
    pred_ctr_y = dx * widths[:, np.newaxis] * np.sin(angles[:, np.newaxis]) + \
                 dy * heights[:, np.newaxis] * np.cos(angles[:, np.newaxis]) + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    # TODO: handle the hard code here
    pred_angle = (norm_angle) * dangle + angles[:, np.newaxis]
    # pred_angle = pred_angle % (2 * np.pi)

    pred_dboxes = np.ones_like(deltas)

    pred_dboxes[:, 0::5] = pred_ctr_x
    pred_dboxes[:, 1::5] = pred_ctr_y
    pred_dboxes[:, 2::5] = pred_w
    pred_dboxes[:, 3::5] = pred_h
    pred_dboxes[:, 4::5] = pred_angle

    return pred_dboxes

def choose_best_match(Rroi, gt_roi):
    """
    chosse best match representation of gt_roi for a Rroi
    :param Rroi: (x_ctr, y_ctr, w, h, angle)
    :param gt_roi: (x_ctr, y_ctr, w, h, angle)
    :return: gt_roi_new: gt_roi with new representation
    """
    # TODO: finish choose best match, this is used for establish map: rotated region feat --> offset
    assert Rroi[4] <= (np.pi)
    assert Rroi[4] >= 0
    assert gt_roi[4] <= ( 2 * np.pi)
    assert gt_roi[4] >= 0

    Rroi_angle = Rroi[4]

    gt_x, gt_y, gt_w, gt_h, gt_angle = gt_roi

    gt_roi_extent = np.array([[gt_x, gt_y, gt_w, gt_h, gt_angle], [gt_x, gt_y, gt_h, gt_w, gt_angle + np.pi/2.],
                              [gt_x, gt_y, gt_w, gt_h, gt_angle + np.pi], [gt_x, gt_y, gt_h, gt_w, gt_angle + np.pi * 3/2.]])

    gt_angle_extent = np.array([gt_angle, gt_angle + np.pi/2., gt_angle + np.pi, gt_angle + np.pi * 3/2.])
    dist = (gt_angle_extent - Rroi_angle) % (2 * np.pi)
    dist = np.min(dist, np.pi * 2 - dist)
    min_index = np.argmin(dist)

    gt_roi_new = gt_roi_extent[min_index]

    return gt_roi_new

def choose_best_match_batch(Rrois, gt_rois):
    """
    chosse best match representation of gt_rois for a Rrois
    :param Rrois: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :param gt_rois: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :return: gt_roi_news: gt_roi with new representation
            shape: (n, 5)
    """
    # assert np.all(Rrois[:, 4] <= np.pi)
    # assert np.all(Rrois[:, 4] >= 0)
    # assert np.all(gt_rois[:, 4] <= (2 * np.pi))
    # assert np.all(gt_rois[:, 4] >= 0)

    # shape: (n, 1)
    Rroi_anlges = Rrois[:, 4][:, np.newaxis]

    gt_xs, gt_ys, gt_ws, gt_hs, gt_angles = copy.deepcopy(gt_rois[:, 0]), copy.deepcopy(gt_rois[:, 1]), \
                                            copy.deepcopy(gt_rois[:, 2]), copy.deepcopy(gt_rois[:, 3]), \
                                            copy.deepcopy(gt_rois[:, 4])
    # shape: (n, 4)
    gt_angle_extent = np.concatenate((gt_angles[:, np.newaxis], (gt_angles + np.pi/2.)[:, np.newaxis],
                                      (gt_angles + np.pi)[:, np.newaxis], (gt_angles + np.pi * 3/2.)[:, np.newaxis]), axis=1)
    dist = (Rroi_anlges - gt_angle_extent) % (2 * np.pi)
    dist = np.minimum(dist, np.pi * 2 - dist)
    min_index = np.argmin(dist, axis=1)
    # selected_index = np.concatenate((np.arange(len(min_index)).reshape(len(min_index), 1), min_index), 1)
    #
    gt_rois_extent0 = copy.deepcopy(gt_rois)
    gt_rois_extent1 = np.hstack((gt_xs[:, np.newaxis], gt_ys[:, np.newaxis], \
                                 gt_hs[:, np.newaxis], gt_ws[:, np.newaxis], gt_angles[:, np.newaxis] + np.pi/2.))
    gt_rois_extent2 = np.hstack((gt_xs[:, np.newaxis], gt_ys[:, np.newaxis], \
                                 gt_ws[:, np.newaxis], gt_hs[:, np.newaxis], gt_angles[:, np.newaxis] + np.pi))
    gt_rois_extent3 = np.hstack((gt_xs[:, np.newaxis], gt_ys[:, np.newaxis], \
                                 gt_hs[:, np.newaxis], gt_ws[:, np.newaxis], gt_angles[:, np.newaxis] + np.pi * 3/2.))

    gt_rois_extent = np.concatenate((gt_rois_extent0[:, np.newaxis, :],
                                     gt_rois_extent1[:, np.newaxis, :],
                                     gt_rois_extent2[:, np.newaxis, :],
                                     gt_rois_extent3[:, np.newaxis, :]), axis=1)

    gt_rois_new = np.zeros_like(gt_rois)
    # pdb.set_trace()
    # TODO: add pool.map here
    for curiter, index in enumerate(min_index):
        gt_rois_new[curiter, :] = gt_rois_extent[curiter, index, :]

    gt_rois_new[:, 4] = gt_rois_new[:, 4] % (2 * np.pi)

    return gt_rois_new

def choose_bset_Rroi_grad_batch(Rroi):
    """
    grad version
    :param Rroi:
    :return:
    """
    x_ctr, y_ctr, w, h, angle = copy.deepcopy(Rroi[:, 0]), copy.deepcopy(Rroi[:, 1]),\
                             copy.deepcopy(Rroi[:, 2]), copy.deepcopy(Rroi[:, 3]), copy.deepcopy(Rroi[:, 4])
    indexes = w < h

    Rroi[indexes, 2] = h[indexes]
    Rroi[indexes, 3] = w[indexes]
    Rroi[indexes, 4] = Rroi[indexes, 4] + np.pi/2.
    Rroi[:, 4] = Rroi[:, 4] % np.pi

    return Rroi, indexes


def choose_best_Rroi_batch(Rroi):
    """
    There are many instances with large aspect ratio, so we choose the point, previous is long side, after is short side, so it makes sure h < w
    then angle % 180,
    :param Rroi: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :return: Rroi_new: Rroi with new representation
    """
    x_ctr, y_ctr, w, h, angle = copy.deepcopy(Rroi[:, 0]), copy.deepcopy(Rroi[:, 1]),\
                             copy.deepcopy(Rroi[:, 2]), copy.deepcopy(Rroi[:, 3]), copy.deepcopy(Rroi[:, 4])
    indexes = w < h

    Rroi[indexes, 2] = h[indexes]
    Rroi[indexes, 3] = w[indexes]
    Rroi[indexes, 4] = Rroi[indexes, 4] + np.pi/2.
    Rroi[:, 4] = Rroi[:, 4] % np.pi

    return Rroi

def dbbox_transform2_best_match_warp(Rrois, gt_boxes):
    """
    :param Rrois: (x_ctr, y_ctr, w, h, angle) angle range in [0, 180]
            (n, 5)
    :param gt_boxes: (x_ctr, y_ctr, w, h, angle) angle range in [0, 360]
            (n, 5)
    :return:
    """

    #  This is a simplified version
    # TODO: preprocess angle of gt_boxes according to the catogeries
    # Here, use a choose beste match angle, similar to choose best point instead
    gt_boxes_new = choose_best_match_batch(Rrois, gt_boxes)
    try:
        assert np.all(Rrois[:, 4] <= (np.pi + 0.001 ))
    except:
        pdb.set_trace()
    bbox_targets = dbbox_transform2_new(Rrois, gt_boxes_new)

    return bbox_targets

def dbbox_transform2_nd(ex_drois, gt_rois):
    """

    :param ex_drois:
    :param gt_rois:
    :return:
    """
    gt_widths = gt_rois[:, 2]
    gt_heights = gt_rois[:, 3]
    gt_angle = gt_rois[:, 4]

    ex_widths = ex_drois[:, 2]
    ex_heights = ex_drois[:, 3]
    ex_angle = ex_drois[:, 4]

    coord = gt_rois[:, 0: 2] - ex_drois[:, 0:2]
    targets_dx = (mx.nd.cos(ex_drois[:, 4]) * coord[:, 0] + mx.nd.sin(ex_drois[:, 4]) * coord[:, 1]) / ex_widths
    targets_dy = (-mx.nd.sin(ex_drois[:, 4]) * coord[:, 0] + mx.nd.cos(ex_drois[:, 4]) * coord[:, 1]) / ex_heights
    targets_dw = mx.nd.log(gt_widths / ex_widths)
    targets_dh = mx.nd.log(gt_heights / ex_heights)
    targets_dangle = (gt_angle - ex_angle) % (2 * np.pi) / (2 * np.pi)
    targets = mx.nd.concat(targets_dx.expand_dims(1),
                            targets_dy.expand_dims(1),
                            targets_dw.expand_dims(1),
                            targets_dh.expand_dims(1),
                            targets_dangle.expand_dims(1), dim=1)
    # targets = mx.nd.stack(targets_dx, targets_dy, targets_dw, targets_dh, targets_dangle, axis=1)

    return targets

def dbbox_transform2_inv(ex_drois, deltas):
    """
    inspired from light-head rcnn, different classes share the same bbox regression
    :param ex_rois: (x, y, w, h, theta) shape (n, 5)
    :param deltas: (dx, dy, dw, dh, dtheta, dx ...) (n, 5 * numclasses)
    :return:
    """
    widths = ex_drois[:, 2]
    heights = ex_drois[:, 3]
    angles = ex_drois[:, 4]
    ctr_x = ex_drois[:, 0]
    ctr_y = ex_drois[:, 1]

    dx = deltas[:, 0::5]
    dy = deltas[:, 1::5]
    dw = deltas[:, 2::5]
    dh = deltas[:, 3::5]

    # clip dw
    # pdb.set_trace()
    # fuck, I write maximum at first
    dw = np.minimum(dw, 4)
    dh = np.minimum(dh, 4)

    dangle = deltas[:, 4::5]
    # pdb.set_trace()
    pred_ctr_x = dx * widths[:, np.newaxis] * np.cos(angles[:, np.newaxis]) \
                - dy * heights[:, np.newaxis] * np.sin(angles[:, np.newaxis]) + ctr_x[:, np.newaxis]
    pred_ctr_y = dx * widths[:, np.newaxis] * np.sin(angles[:, np.newaxis]) + \
                dy * heights[:, np.newaxis] * np.cos(angles[:, np.newaxis]) + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    # pred_ctr_x = dx * widths * np.cos(angles) \
    #             - dy * heights * np.sin(angles) + ctr_x
    # pred_ctr_y = dx * widths * np.sin(angles) + \
    #             dy * heights * np.cos(angles) + ctr_y
    # pred_w = np.exp(dw) * widths
    # pred_h = np.exp(dh) * heights

    # TODO: handle the hard code here
    pred_angle = (2 * np.pi) * dangle + angles[:, np.newaxis]
    pred_angle = pred_angle % ( 2 * np.pi)

    pred_dboxes = np.ones_like(deltas)

    pred_dboxes[:, 0::5] = pred_ctr_x
    pred_dboxes[:, 1::5] = pred_ctr_y
    pred_dboxes[:, 2::5] = pred_w
    pred_dboxes[:, 3::5] = pred_h
    pred_dboxes[:, 4::5] = pred_angle

    return pred_dboxes

def bbox_overlaps_py(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps

def clip_polys(polys, im_shape):
    """
    clip boxes to image boundaries
    :param polys: [N, 8 * num_classes]
    :param im_shape: tuple of 2
    :return: [N, 8 * num_classes]
    """
    def clip_wh(coords, wh):
        return np.maximum(np.minimum(coords, wh - 1), 0)

    # im_shape[1] > x1 >= 0
    polys[:, 0::8] = np.maximum(np.minimum(polys[:, 0::8], im_shape[1] - 1), 0)
    # im_shape[0] > y1 >= 0
    polys[:, 1::8] = np.maximum(np.minimum(polys[:, 1::8], im_shape[0] - 1), 0)
    # 0 <= x2 < im_shape[1]
    polys[:, 2::8] = np.maximum(np.minimum(polys[:, 2::8], im_shape[1] - 1), 0)
    # 0 <= y2 < im_shape[0]
    polys[:, 3::8] = np.maximum(np.minimum(polys[:, 3::8], im_shape[0] - 1), 0)

    polys[:, 4::8] = clip_wh(polys[:, 4::8], im_shape[1])

    polys[:, 5::8] = clip_wh(polys[:, 5::8], im_shape[0])

    polys[:, 6::8] = clip_wh(polys[:, 6::8], im_shape[1])

    polys[:, 7::8] = clip_wh(polys[:, 7::8], im_shape[0])

    return polys
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def filter_boxes(boxes, min_size):
    """
    filter small boxes.
    :param boxes: [N, 4* num_classes]
    :param min_size:
    :return: keep:
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def nonlinear_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)
    # print 'in nonlinear_transform'
    # pdb.set_trace()
    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_wrapp(coordinate):
    coordinate = np.array(coordinate).reshape(4, 2)
    output = get_best_begin_point(coordinate)
    output = np.array(output).reshape(8)
    return output
def get_best_begin_point(coordinate):
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        print("choose one direction!")
    return  combinate[force_flag]

def nonlinear_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]

    pred_w = np.clip(np.exp(dw), -60, 60) * widths[:, np.newaxis]
    pred_h = np.clip(np.exp(dh), -60, 60) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes


def iou_transform(ex_rois, gt_rois):
    """ return bbox targets, IoU loss uses gt_rois as gt """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'
    return gt_rois


def iou_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    dx1 = box_deltas[:, 0::4]
    dy1 = box_deltas[:, 1::4]
    dx2 = box_deltas[:, 2::4]
    dy2 = box_deltas[:, 3::4]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = dx1 + x1[:, np.newaxis]
    # y1
    pred_boxes[:, 1::4] = dy1 + y1[:, np.newaxis]
    # x2
    pred_boxes[:, 2::4] = dx2 + x2[:, np.newaxis]
    # y2
    pred_boxes[:, 3::4] = dy2 + y2[:, np.newaxis]

    return pred_boxes


# define bbox_transform and bbox_pred
bbox_transform = nonlinear_transform
bbox_pred = nonlinear_pred
