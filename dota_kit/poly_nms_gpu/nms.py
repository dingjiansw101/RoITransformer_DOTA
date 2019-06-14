import numpy as np

from poly_nms import poly_gpu_nms
from poly_overlaps import poly_overlaps

def poly_gpu_nms_wrapper(thresh, device_id):
    def _nms(dets):
        return poly_gpu_nms(dets, thresh, device_id)
    return _nms

def poly_overlaps_nms_wrapper(device_id):
    def _overlaps(boxes, query_boxes):
        return poly_overlaps(boxes, query_boxes, device_id)
    return _overlaps