# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yi Li, Haochen Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fpn/cfgs/resnet_v1_101_dota_rotbox_light_head_RoITransformer_trainval_fpn_end2end.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
sys.path.insert(0, '../')
import mxnet as mx
from core.tester import im_detect, Predictor, im_detect_rotbox_Rroi
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from dota_kit.ResultMerge import py_cpu_nms_poly
from utils import image
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use fpn only (w/o Deformable ConvNets)', default=False, action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()

def draw_all_poly_detection(im_array, detections, class_names, scale, cfg, threshold=0.2):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    # pdb.set_trace()
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        try:
            dets = detections[j]
        except:
            pdb.set_trace()
        for det in dets:
            bbox = det[:8] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            # draw first point
            cv2.circle(im, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
            for i in range(3):
                cv2.line(im, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2)
            cv2.line(im, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def main():
    # get symbol
    pprint.pprint(config)
    # config.symbol = 'resnet_v1_101_rfcn_dcn' if not args.rfcn_only else 'resnet_v1_101_rfcn'
    config.symbol = 'resnet_v1_101_fpn_rcnn_rotbox_light_head_RoITransformer'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 15
    classes = ['__background__',  # always index 0
                        'plane', 'baseball-diamond',
                        'bridge', 'ground-track-field',
                        'small-vehicle', 'large-vehicle',
                        'ship', 'tennis-court',
                        'basketball-court', 'storage-tank',
                        'soccer-ball-field', 'roundabout',
                        'harbor', 'swimming-pool',
                        'helicopter']
    # load demo data
    image_names = ['P0004__1__0___0.png', 'P0053__1__0___0.png', 'P0060__1__1648___824.png']
    data = []
    for im_name in image_names:
        # pdb.set_trace()
        assert os.path.exists(cur_path + '/../demo/' + im_name), ('%s does not exist'.format('../demo/' + im_name))
        im = cv2.imread(cur_path + '/../demo/' + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})


    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    # arg_params, aux_params = load_param(cur_path + '/../model/' + ('rfcn_dcn_coco' if not args.rfcn_only else 'rfcn_coco'), 0, process=True)
    # TODO: change this path
    arg_params, aux_params = load_param(r'/home/dj/code/Deformable_FPN_DOTA/output/fpn/DOTA/resnet_v1_101_dota_rotbox_light_head_Rroi_v6_trainval_fpn_end2end/train/fpn_DOTA_oriented',
                                            config.TEST.test_epoch, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    # warm up
    for j in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        scores, boxes, data_dict = im_detect_rotbox_Rroi(predictor, data_batch, data_names, scales, config)

    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, data_dict = im_detect_rotbox_Rroi(predictor, data_batch, data_names, scales, config)
        # boxes = boxes[0].astype('f')
        # scores = scores[0].astype('f')
        boxes = boxes[0].astype('float64')
        scores = scores[0].astype('float64')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            # cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_boxes = boxes[:, 8:16] if config.CLASS_AGNOSTIC else boxes[:, j * 8:(j + 1) * 8]
            cls_quadrangle_dets = np.hstack((cls_boxes, cls_scores))
            # keep = nms(cls_dets)
            keep = py_cpu_nms_poly(cls_quadrangle_dets, 0.3)
            cls_quadrangle_dets = cls_quadrangle_dets[keep, :]
            cls_quadrangle_dets = cls_quadrangle_dets[cls_quadrangle_dets[:, -1] > 0.7, :]
            dets_nms.append(cls_quadrangle_dets)
        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        # im = cv2.imread(cur_path + '/../demo/' + im_name)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # pdb.set_trace()
        im = draw_all_poly_detection(data_dict[0]['data'].asnumpy(), dets_nms, classes[1:], data[idx][1].asnumpy()[0][2], config,
                                     threshold=0.2)
        cv2.imwrite(cur_path + '/../demo/' + 'results' + im_name, im)
        # show_boxes(im, dets_nms, classes, 1)

    print 'done'

if __name__ == '__main__':
    main()
