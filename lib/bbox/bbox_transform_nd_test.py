import unittest
import numpy as np
# from bbox_transform_nd import *
from bbox_transform import *
import mxnet as mx
import copy

class Testbbox_transform(unittest.TestCase):


    def test_bbox_poly2hbb(self):
        polys = np.array([[1, 3, 21, 3, 21, 83, 1, 83, 1],
                             [50, 100, 65, 100, 65, 145, 50, 145, 3]])
        # expected = np.array([[11, 43, 20, 80],
        #                      [115/2.0, 245/2.0, 15, 45]])
        expected = np.array([[1, 3, 21, 83, 1],
                             [50, 100, 65, 145, 3]])
        polys = mx.nd.array(polys)
        expected = mx.nd.array(expected)
        output = bbox_poly2hbb_nd(polys)
        np.testing.assert_array_almost_equal(output.asnumpy(), expected.asnumpy())

    def test_dbbox_transform2_nd(self):
        """
        encoding format similar to RRPN, except the angle was restricted to [0, 2 pi], dangle was restricted to [0, 1]

        Must Test corner cases
        :return:
        """
        # TODO: the bbox_means need to be changed
        boxlist1 = np.array([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 100, 60, 34, np.pi/2]
                                  ])
        boxlist2 = np.array([[1, 1, 5, 8, np.pi/16],
                                  [1, 1, 5, 8, np.pi/16 + np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 90, 12, 45, np.pi/10]
                                  ])
        # TODO: check the case by hand
        expected_targets = np.array([[0.0000,  0.0000, -0.6931,  0.4700,  0.0312],
                                        [0.0000,  0.0000, -0.6931,  0.4700,  0.0313],
                                        [0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                        [-0.1667,  0.0000, -1.6094,  0.2803, 0.8]] )
        output = dbbox_transform2(boxlist1, boxlist2)
        np.testing.assert_almost_equal(expected_targets, output, decimal=4)
        expected_targets_nd = dbbox_transform2_nd(mx.nd.array(boxlist1, dtype='float32'), mx.nd.array(boxlist2, dtype='float32'))

        np.testing.assert_almost_equal(expected_targets, expected_targets_nd.asnumpy(), decimal=4)

    @unittest.skip("The test need to be reconstruct")
    def test_dbbox_transform2_warp(self):
        # boxlist1 = np.array([[1, 1, 10, 5, 0],
        #                           [1, 1, 10, 5, np.pi/10],
        #                           [1, 1, 10, 5, 0],
        #                           [30, 100, 60, 34, np.pi/2]
        #                           ])
        # TODO: add corner cases
        boxlist1 = np.array([[-1, -2.5, 3, 4.5],
                             [24.5, 68.0, 35.5, 112.0],
                             # [2, 4, 24.6, 8],
                             [-9.8, 0.5, 13.8, 7.5]
                             ]) # (xmin, ymin, xmax, ymax)
        boxlist2 = np.array([[1, 1, 5, 8, np.pi/16],
                                  # [1, 1, 5, 8, np.pi/16 + np.pi/10]
                                  # [1, 1, 10, 5, 0],
                                  [30, 90, 12, 45, np.pi/10],
                             [5, 4, 26, 8.2, np.pi/2 + np.pi/10.]
                                  ])
        polys2 = RotBox2Polys(boxlist2)
        targets = dbbox_transform2_warp(boxlist1, polys2)
        expected_targets = np.array([[0, 0, 0, 0, 0.78125],
                                     [0, 0, 0, 0, 0.8],
                                     [0., -3/8., np.log(26/24.6), np.log(8.2/8.), np.pi/10./(2 * np.pi)]])
        np.testing.assert_almost_equal(expected_targets, targets, decimal=4)

        targets_nd = dbbox_transform2_warp_nd(mx.nd.array(boxlist1), mx.nd.array(polys2))
        np.testing.assert_almost_equal(expected_targets, targets_nd.asnumpy(), decimal=4)
if __name__ == '__main__':
    unittest.main()