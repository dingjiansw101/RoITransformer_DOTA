import unittest
import numpy as np
from bbox_regression import *
import sys
sys.path.insert(0, '../../fpn')
from config.config import config as cfg



class Testbbox_regression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg.CLASS_AGNOSTIC = False
        cfg.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])
    def test_expand_bbox_regression_targets(self):
        pass

    def test_expand_bbox_regression_targets_base_4(self):
        # [k * (p + 1)] --> [k * p * num_classes]
        # [k * 5] --> [k * 4 * num_classes]
        num_classes = 3
        bbox_targets_data = np.array([[0, 3, 4, 10, 20],
                                      [2, 7, 11, 23, 2]])

        expected_targets = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 7, 11, 23, 2]])
        expected_weights = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

        calc_targets, calc_weights = expand_bbox_regression_targets(bbox_targets_data, num_classes, cfg)
        calc_targets2, calc_weights2 = expand_bbox_regression_targets_base(bbox_targets_data, num_classes, cfg)

        np.testing.assert_array_almost_equal(calc_targets, expected_targets)
        np.testing.assert_array_almost_equal(calc_weights, expected_weights)

        np.testing.assert_array_almost_equal(calc_targets2, expected_targets)
        np.testing.assert_array_almost_equal(calc_weights2, expected_weights)

    def test_expand_bbox_regression_targets_base_8(self):
        # [k * (p + 1)] --> [k * p * num_classes]
        # [k * 9] --> [k * 8 * num_classes]
        num_classes = 3
        # (dx1, dy1, ... dx4, dy4)
        bbox_targets_data = np.array([[1, 3, 4, 2, 2, 2, 1, 9, 1],
                                      [0, 2, 2, 3, 1, 3, 6, 3, 1]])
        expected_targets = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 2, 2, 1, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]])
        expected_weights = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        calc_targets, calc_weights = expand_bbox_regression_targets_base(bbox_targets_data, num_classes, cfg)

        np.testing.assert_array_almost_equal(calc_targets, expected_targets)
        np.testing.assert_array_almost_equal(calc_weights, expected_weights)

if __name__ == '__main__':
    unittest.main()