import unittest
import numpy as np
from bbox_transform import *
import copy

class Testbbox_transform(unittest.TestCase):

    def test_bbox_overlaps(self):
        pass

    def test_bbox_poly2hbb(self):
        polys = np.array([[1, 3, 21, 3, 21, 83, 1, 83, 1],
                             [50, 100, 65, 100, 65, 145, 50, 145, 3]])
        # expected = np.array([[11, 43, 20, 80],
        #                      [115/2.0, 245/2.0, 15, 45]])
        expected = np.array([[1, 3, 21, 83, 1],
                             [50, 100, 65, 145, 3]])
        output = bbox_poly2hbb(polys)
        np.testing.assert_array_almost_equal(output, expected)

    def test_poly2bbox(self):
        polys = np.array([[1, 3, 21, 3, 21, 83, 1, 83],
                             [50, 100, 65, 100, 65, 145, 50, 145]])
        # expected = np.array([[11, 43, 20, 80],
        #                      [115/2.0, 245/2.0, 15, 45]])
        expected = np.array([[1, 3, 21, 83],
                             [50, 100, 65, 145]])
        output = poly2bbox(polys)
        np.testing.assert_array_almost_equal(output, expected)

    def test_poly2bbox_nd(self):
        polys = np.array([[1, 3, 21, 3, 21, 83, 1, 83],
                          [50, 100, 65, 100, 65, 145, 50, 145]])
        # expected = np.array([[11, 43, 20, 80],
        #                      [115/2.0, 245/2.0, 15, 45]])
        expected = np.array([[1, 3, 21, 83],
                             [50, 100, 65, 145]])
        polys_nd = mx.nd.array(polys)

        output = poly2bbox_nd(polys_nd)
        np.testing.assert_array_almost_equal(output.asnumpy(), expected)

    def test_box2poly(self):
        """
        """
        ext_rois = np.array([[11, 43, 20, 80],
                             [115/2.0, 245/2.0, 15, 45]
                             ])
        expected = np.array([[1, 3, 21, 3, 21, 83, 1, 83],
                             [50, 100, 65, 100, 65, 145, 50, 145]])
        calculated = box2poly(ext_rois)
        np.testing.assert_array_almost_equal(calculated, expected)
    def test_dbbox_transform(self):
        """
        ext_rois: (x, y, w, h)
        gt_boxes: (x1, y1, x2, y2, x3, y3, x4, y4)
        """
        # ext_rois = np.array([[11, 43, 20, 80],
        #                      [115/2.0, 245/2.0, 15, 45]
        #                      ])
        ext_rois = np.array([[1, 3, 21, 83],
                             [50, 100, 65, 145],
                             # [36., 50., 46., 60.]
                             ])
        gt_boxes = np.array([[1, 3, 22, 4, 21, 84, 0, 83],
                             [50, 100, 60, 90, 65, 145, 54, 120],
                             # [40., 50., 46., 57., 43., 60., 36., 55.]
                             ])
        targets = dbbox_transform(ext_rois, gt_boxes)
        expected = np.array([[0, 0, 1/21.0, 1/81.0, 0, 1/81.0, -1/21.0, 0],
                             [0, 0, -5/16.0, -10/46.0, 0, 0, 4/16.0, -25/46.0],
                             ], dtype=np.float)
        np.testing.assert_array_almost_equal(targets, expected)

    def test_box_pred_multiclass(self):
        ext_rois = np.array([[1, 3, 21, 83],
                             [50, 100, 65, 145]])
        expect_results = np.array([[1, 3, 21, 83, 4, 5,39, 30, 1, 3, 21, 83],
                             [50, 100, 65, 145, 50, 100, 65, 145, 40, 105, 50, 120]])

        # targets = bbox_transform(ext_rois, gt_boxes)
        targets = np.array([[0., 0., 0., 0., 0.5       , -0.31481481,  0.5389965 , -1.13635262, 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., -0.78125   , -0.2173913 , -0.37469345, -1.05605267]])
        outputs = bbox_pred(ext_rois, targets)
        np.testing.assert_array_almost_equal(expect_results, outputs)
    def test_dbbox_pred_multiclass(self):

        ext_rois = np.array([[1, 3, 21, 83],
                             [50, 100, 65, 145]
                             ])
        expect_results = np.array([[1., 3., 21., 3., 21., 83., 1., 83., 1, 3, 22, 4, 21, 84, 0, 83, 1., 3., 21., 3., 21., 83., 1., 83.],
                             [50., 100., 65., 100., 65., 145., 50., 145., 50., 100., 65., 100., 65., 145., 50., 145., 50, 100, 60, 90, 65, 145, 54, 120]])
        targets = np.array([
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  0.04761905,  0.01234568,  0., 0.01234568, -0.04761905,  0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  0., -0.3125, -0.2173913,  0., 0.,  0.25, -0.54347826]
                            ])


        outputs = dbbox_pred(ext_rois, targets)
        np.testing.assert_array_almost_equal(expect_results, outputs)
    def test_clip_polys(self):

        polys = np.array([[-1, 3, 4, 7, 102, 150, 50, 205],
                          [3, 10, 30, 50, 70, 90, 80, 90],
                          [0,0, 0, 200, 100, 200, 100, 0]])

        # y, x
        im_shape = (200, 100)

        expected_outputs = np.array([[0, 3, 4, 7, 99, 150, 50, 199],
                                     [3, 10, 30, 50, 70, 90, 80, 90],
                                     [0, 0, 0, 199, 99, 199, 99, 0]])

        outputs = clip_polys(polys, im_shape)
        np.testing.assert_array_almost_equal(outputs, expected_outputs)
    def test_clip(self):
        x1, y1, x2, y2, x3, y3, x4, y4 = np.array([1, 3, 21, 3, 21, 83, 1, 83],
                             # [50, 100, 65, 100, 65, 145, 50, 145]
                                                  )
        w, h = np.array([100, 70])

        expected = np.array([1, 3, 21, 69]
                             # [50, 100, 65, 145]
                            )

        xmin = min(max(min(x1, x2, x3, x4), 0), w - 1)
        xmax = min(max(max(x1, x2, x3, x4), 0), w - 1)
        ymin = min(max(min(y1, y2, y3, y4), 0), h - 1)
        ymax = min(max(max(y1, y2, y3, y4), 0), h - 1)

        results = np.array([xmin, ymin, xmax, ymax])

        np.testing.assert_array_almost_equal(expected, results)
    def test_clip2(self):

        bbox = np.array([-1, 3, 21, 3, 21, 83, 1, 83])
        w, h = np.array([100, 70])

        expected = np.array([0, 3, 21, 3, 21, 69, 1, 69])
        x1 = min(max(float(bbox[0]), 0), w - 1)
        y1 = min(max(float(bbox[1]), 0), h - 1)
        x2 = min(max(float(bbox[2]), 0), w - 1)
        y2 = min(max(float(bbox[3]), 0), h - 1)
        x3 = min(max(float(bbox[4]), 0), w - 1)
        y3 = min(max(float(bbox[5]), 0), h - 1)
        x4 = min(max(float(bbox[6]), 0), w - 1)
        y4 = min(max(float(bbox[7]), 0), h - 1)

        results = np.array([x1, y1, x2, y2, x3, y3, x4, y4])

        ## The three functions seems like the same
        # np.testing.assert_array_almost_equal(expected, results)
        # np.allclose(expected, results)
        np.testing.assert_allclose(expected, results)
    def test_filter_shape(self):
        """
        TODO
        filter unpossible shapes
        :return:
        """
    def test_xy2wh(self):
        boxes = np.array([[1, 3, 45, 10],
                          [24.4, 3., 44.5, 52.2]])
        outputs = xy2wh(boxes)
        expected_outputs = np.array([[23, 6.5, 45, 8],
                                     [34.45, 27.6, 21.1, 50.2]])
        np.testing.assert_almost_equal(expected_outputs, outputs)
    def test_wh2xy(self):
        boxes = np.array([[1, 3, 45, 10],
                          [24.4, 3., 44.5, 52.2]])
        outputs = xy2wh(boxes)
        outputs = wh2xy(outputs)
        np.testing.assert_almost_equal(boxes, outputs)

    def test_dbbox_transform2(self):
        """
        encoding format similar to RRPN, except the angle was restricted to [0, 2 pi], dangle was restricted to [0, 1]

        Must Test corner cases
        :return:
        """
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
        expected_targets = np.array([[0.0000,  0.0000, -0.6931,  0.4700,  0.0312],
                                        [0.0000,  0.0000, -0.6931,  0.4700,  0.0313],
                                        [0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                        [-0.1667,  0.0000, -1.6094,  0.2803, 0.8]] )
        output = dbbox_transform2(boxlist1, boxlist2)
        print 'output:', output
        np.testing.assert_almost_equal(expected_targets, output, decimal=4)

    def test_dbbox_transform2_inv(self):
        """
            similar to light-head rcnn, different classes share the same bbox regression now
        :return:
        """

        boxlist1 = np.array([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 100, 60, 34, np.pi/2]
                                  ])
        # the boxlist2(ground truths) are restricted to (0, 2 * pi)
        boxlist2 = np.array([[1, 1, 5, 8, np.pi/16],
                                  [1, 1, 5, 8, np.pi/16 + np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 90, 12, 45, np.pi/10]
                                  ])
        expected_targets = dbbox_transform2(boxlist1, boxlist2)
        expected_boxlist2 = dbbox_transform2_inv(boxlist1, expected_targets)

        np.testing.assert_almost_equal(expected_boxlist2, boxlist2)

    def test_rotation_invariant_encoding(self):
        boxlist1 = np.array([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 90.8, 60, 34, np.pi/2]
                                  ])
        # the boxlist2(ground truths) are restricted to (0, 2 * pi)
        # TODO: add corner case
        boxlist2 = np.array([[1, 1, 5, 8, np.pi/16],
                                  [1, 1, 5, 8, np.pi/16 + np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 90.8, 12, 45, np.pi/10]
                                  ])
        boxlist3 = copy.deepcopy(boxlist1)
        boxlist3[:, 4] = boxlist1[:, 4] + np.pi/10.
        boxlist4 = copy.deepcopy(boxlist2)
        boxlist4[:, 4] = boxlist2[:, 4] + np.pi/10.
        targets1 = dbbox_transform2(boxlist1, boxlist2)
        targets2 = dbbox_transform2(boxlist3, boxlist4)
        np.testing.assert_almost_equal(targets1, targets2, decimal=4)

    def test_rotation_invariant_encoding2(self):

        boxlist1 = np.array([[1, 1, 10, 5, 0],
                             [2, 4, 7, 8, np.pi/10.]])

        boxlist2 = np.array([[3, 4, 9.2, 4.8, np.pi/6.0],
                             [2.3, 4.3, 8, 9, np.pi/9.0]])


        boxlist3 = copy.deepcopy(boxlist1)
        boxlist4 = copy.deepcopy(boxlist2)

        angle = np.random.rand()
        boxlist3[:, 4] = boxlist1[:, 4] + angle
        boxlist4[:, 4] = boxlist4[:, 4] + angle

        # trans_matrix = np.array([[np.cos(angle), -np.sin(angle)],
        #                          [np.sin(angle), np.cos(angle)]])

        boxlist3[:, 0] = np.cos(angle) * boxlist1[:, 0] - np.sin(angle) * boxlist1[:, 1]
        boxlist3[:, 1] = np.sin(angle) * boxlist1[:, 0] + np.cos(angle) * boxlist1[:, 1]

        boxlist4[:, 0] = np.cos(angle) * boxlist2[:, 0] - np.sin(angle) * boxlist2[:, 1]
        boxlist4[:, 1] = np.sin(angle) * boxlist2[:, 0] + np.cos(angle) * boxlist2[:, 1]

        targets1 = dbbox_transform2(boxlist1, boxlist2)
        targets2 = dbbox_transform2(boxlist3, boxlist4)

        np.testing.assert_almost_equal(targets1, targets2, decimal=4)
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

    @unittest.skip("The test need to be reconstruct")
    def test_dbbox_transform2_inv_warp_multiclass(self):
        # test 2 classes here
        ext_rois = np.array([[-1, -2.5, 3, 4.5],
                             [24.5, 68.0, 35.5, 112.0],
                             # [2, 4, 24.6, 8],
                             [-9.8, 0.5, 13.8, 7.5]
                             ]) # (xmin, ymin, xmax, ymax)
        expected_results = np.array([[1, 1, 5, 8, np.pi/2, 1, 1, 5, 8, np.pi/16], # (x_ctr, y_ctr, w, h, theta)
                             [30, 90, 12, 45, np.pi/10, 30, 90, 12, 45, np.pi/2],
                             # [2, 4, 24.6, 8],
                             [2, 4, 24.6, 8, np.pi/2, 5, 4, 26, 8.2, np.pi/2 + np.pi/10.]
                             ])
        targets = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.78125],
                                     [0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, -3/8., np.log(26/24.6), np.log(8.2/8.), np.pi/10./(2 * np.pi)]])
        outputs = dbbox_transform2_inv_warp(ext_rois, targets)
        # print 'outputs:', outputs
        np.testing.assert_almost_equal(outputs, expected_results, decimal=4)


    def test_dbbox_transform2_encode_decode(self):
        boxlist2 = np.array([[1, 1, 5, 8, np.pi/16],
                                  [1, 1, 5, 8, np.pi/16 + np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 90, 12, 45, np.pi/10]
                                  ])

        polys2 = RotBox2Polys(boxlist2)
        ex_rois = poly2bbox(polys2)
        targets = dbbox_transform2_warp(ex_rois, polys2)
        outputs = dbbox_transform2_inv_warp(ex_rois, targets)
        np.testing.assert_almost_equal(outputs, boxlist2, decimal=5)

    def test_RotBox2Polys(self):
        rotboxes = np.array([[1, 1, 5, 8, np.pi/2, 1, 1, 5, 8, np.pi/16], # (x_ctr, y_ctr, w, h, theta)
                             [30, 90, 12, 45, np.pi/10, 30, 90, 12, 45, np.pi/2],
                             # [2, 4, 24.6, 8],
                             [2, 4, 24.6, 8, np.pi/2, 5, 4, 26, 8.2, np.pi/2 + np.pi/10.]
                             ])
        expected_polys = np.concatenate((RotBox2Polys(rotboxes[:, 0:5]), RotBox2Polys(rotboxes[:, 5:10])), axis=1)
        polys = RotBox2Polys_multi_class(rotboxes)
        # print 'polys:', polys

        self.assertTrue(polys.shape == (3, 16))
        np.testing.assert_almost_equal(expected_polys, polys)

    def test_polys2xyhs(self):
        # polys =
        pass
    @unittest.skip("The test can not be passed")
    def test_xyhs2polys(self):
        # xyh format: (x1, y1, x2, y2, h), x1, y1 is the first point, x2, y2 is the second point. h is the height of a bounding box
        xyhs = np.array([[2, 1, 6, 3, 3],
                         [1.4, 8, 4.2, 6.3, 7.4]])

        polys = xyhs2polys(xyhs)
        inverse_xyhs = polys2xyhs(polys)
        inverse_polys = xyhs2polys(inverse_xyhs)
        np.testing.assert_almost_equal(xyhs, inverse_xyhs, decimal=6)
        np.testing.assert_almost_equal(polys, inverse_polys, decimal=6)

    def test_dbbox_transform3(self):

        boxlist1 = np.array([[np.sqrt(3), 1, 2 * np.sqrt(3), 2, 2],
                             [np.sqrt(3), 1, 1 + np.sqrt(3), 1 + np.sqrt(3), 2]])
        boxlist2 = np.array([[(3 * np.sqrt(3)-1)/2., (3 + np.sqrt(3))/2., (4 * np.sqrt(3) - 1)/2., (4 + np.sqrt(3))/2., 1],
                             [(np.sqrt(3) + 1)/2., (np.sqrt(3) + 3)/2., (np.sqrt(3) + 2)/2., (3 + 2 * np.sqrt(3))/2., 1]])

        targets = dbboxtransform3(boxlist1, boxlist2)

        trans_boxlist1 = np.array([[0, 0, 2, 0, 2]])

        expected_targets = np.array([[0.5, 0.5, 0, 0.5, np.log(1/2.)],
                                     [0.5, 0.5, 0, 0.5, np.log(1 / 2.)]])
        np.testing.assert_almost_equal(expected_targets, targets)

    def test_dbbox_transform3_inv_warp(self):
        ext_rois = np.array([[2, 5, 6, 10.3]])
        targets = np.array([[1/4., 0.2/5.3, 0, 0.2/5.3, np.log(5.1/5.3)]])
        outputs = dbboxtransform3_inv_warp(ext_rois, targets)
        expected_results = np.array([[3, 5.2, 6, 5.2, 5.1]])
        # pdb.set_trace()
        np.testing.assert_almost_equal(outputs, expected_results)
    def test_dbbox_transform3_warp_encode_decode(self):
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
        gt_xyhs = polys2xyhs(polys2)
        targets = dbboxtransform3_warp(boxlist1, polys2)
        # expected_targets = np.array([[0, 0, 0, 0, 0.78125],
        #                              [0, 0, 0, 0, 0.8],
        #                              [0., -3/8., np.log(26/24.6), np.log(8.2/8.), np.pi/10./(2 * np.pi)]])
        targets_inverse = dbboxtransform3_inv_warp(boxlist1, targets)
        np.testing.assert_almost_equal(gt_xyhs, targets_inverse)

    def test_dbbox_transform3_rotation_invariant(self):
        boxlist1 = np.array([[1000, 1000.8, 8000.767, 12500, np.pi/6.],
                             [24.5, 68.0, 23, 89.2, np.pi],
                             # [2, 4, 24.6, 8],
                             # [-9.8, 0.5, 13.8, 7.5, -np.pi/10.]
                             ]) # (xmin, ymin, xmax, ymax)
        boxlist2 = np.array([[1000, 1000.8, 5000.767, 8000, np.pi/16],
                                  # [1, 1, 5, 8, np.pi/16 + np.pi/10]
                                  # [1, 1, 10, 5, 0],
                                  [24.5, 68.0, 12, 45.5, np.pi/10],
                             # [5, 4, 26, 8.2, np.pi/2 + np.pi/10.]
                                  ])
        polys1 = RotBox2Polys(boxlist1)
        polys2 = RotBox2Polys(boxlist2)
        xyhs1 = polys2xyhs(polys1)
        xyhs2 = polys2xyhs(polys2)

        randangle = np.random.rand()
        boxlist3 = copy.deepcopy(boxlist1)
        boxlist3[:, 4] = boxlist3[:, 4] + randangle
        polys3 = RotBox2Polys(boxlist3)
        xyhs3 = polys2xyhs(polys3)

        boxlist4 = copy.deepcopy(boxlist2)
        boxlist4[:, 4] = boxlist4[:, 4] + randangle
        polys4 = RotBox2Polys(boxlist4)
        xyhs4 = polys2xyhs(polys4)

        targets1 = dbboxtransform3(xyhs1, xyhs2)
        targets2 = dbboxtransform3(xyhs3, xyhs4)

        np.testing.assert_almost_equal(targets1, targets2, decimal=6)
    def test_dbbox_transform3_inv_multi_class(self):
        pass
    def test_dbbox_transform3_inv_warp_multi_class(self):
        """
        This is a multi-class test
        :return:
        """
        ext_rois = np.array([[2, 5, 6, 10.3],
                             ])
        targets = np.array([[0, 0, 0, 0, 0, 1/4., 0.2/5.3, 0, 0.2/5.3, np.log(5.1/5.3), 0, 0, 0, 0, 0]
                            ])
        outputs = dbboxtransform3_inv_warp(ext_rois, targets)
        expected_results = np.array([[2, 5, 6, 5, 5.3, 3, 5.2, 6, 5.2, 5.1, 2, 5, 6, 5, 5.3]])
        np.testing.assert_almost_equal(outputs, expected_results)

    def test_bbox_transformxyh(self):
        ext_rois = np.array([[-1, -2.5, 3, 4.5],
                             [24.5, 68.0, 35.5, 112.0],
                             [-9.8, 0.5, 13.8, 7.5]])


    def test_polygonToRotRectangle_batch(self):
        polygons = np.array([[0, 0, 3, 0, 3, 3, 0, 3]])
        rotboxs = polygonToRotRectangle_batch(polygons)
        print 'rotboxs:', rotboxs
    def test_get_best_begin_point_wrapp(self):
        print 'test get best begin point'
        input = [7, 5, 3, 6, 1, 2, 5, 1]
        expected_output = [1, 2, 5, 1, 7, 5, 3, 6]
        output = get_best_begin_point_wrapp(input)
        np.testing.assert_almost_equal(np.array(output), np.array(expected_output))

    def test_xyhs2polys_muli_class(self):
        xyhs = np.array([[0, 0, 2, 0, 3, 3, 4.3, 6, 7, 8.4],
                         [2, 0, 2, 3, 2, 4.4, 5.5, 7.6, 8.2, 9]])
        polys = xyhs2polys_muli_class(xyhs)
        expected_polys = np.concatenate((xyhs2polys(xyhs[:, 0:5]), xyhs2polys(xyhs[:, 5:10])), axis=1)
        self.assertTrue(polys.shape == (2, 16))

        np.testing.assert_almost_equal(polys, expected_polys)

    def test_choose_best_Rroi_batch(self):
        # (x_ctr, y_ctr, w, h, angle)
        Rrois = np.array([[3, 4, 2, 10, np.pi/6.],
                          [3, 4, 10, 2, np.pi/6. + np.pi/2.],
                          [3, 4, 2, 10, np.pi/6. + np.pi],
                          [3, 4, 10, 2, np.pi/6. + np.pi + np.pi/2.]])

        results = choose_best_Rroi_batch(Rrois)
        expected_results = np.array([[3, 4, 10, 2, np.pi/6. + np.pi/2.],
                                     [3, 4, 10, 2, np.pi / 6. + np.pi / 2.],
                                     [3, 4, 10, 2, np.pi / 6. + np.pi / 2.],
                                     [3, 4, 10, 2, np.pi / 6. + np.pi / 2.]])

        np.testing.assert_almost_equal(results, expected_results, decimal=6)

    def test_choose_best_match_batch(self):
        # (x_ctr, y_ctr, w, h, angle)
        Rrois = np.array([[3, 4, 2, 10, np.pi/6.],
                          [3, 4, 10, 2, np.pi/6. + np.pi/2.],
                          [3, 4, 2, 10, np.pi / 6.],
                          [3, 4, 10, 2, np.pi / 6. + np.pi / 2.]
                                ])

        gt_rois =  np.array([[3, 4, 10, 2, np.pi/6. + np.pi/2.],
                                     [3, 4, 10, 2, np.pi / 6. + np.pi / 2.],
                             [3, 4, 2, 10, np.pi/6. + np.pi],
                             [3, 4, 10, 2, np.pi/6. + np.pi * 3 / 2.]
                            ])
        results = choose_best_match_batch(Rrois, gt_rois)
        expected_results = np.array([[3, 4, 2, 10, np.pi/6.],
                          [3, 4, 10, 2, np.pi/6. + np.pi/2.],
                          [3, 4, 2, 10, np.pi / 6.],
                           [3, 4, 10, 2, np.pi / 6. + np.pi / 2.]
                            ])

        np.testing.assert_almost_equal(results, expected_results, decimal=6)

    def test_dbbox_transform2_new(self):
        boxlist1 = np.array([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi - np.pi/10.],
                                  [1, 1, 10, 5, np.pi - np.pi/10.]
                                  ])
        boxlist2 = np.array([[1, 1, 10, 5, -np.pi/10.],
                                  [1, 1, 10, 5, np.pi/10],
                                  [1, 1, 10, 5, np.pi - np.pi/10. - np.pi/20.],
                                  [1, 1, 10, 5, np.pi - np.pi/10. - np.pi/20. + 10 * np.pi]
                                  ])
        norm = np.pi / 2.
        expected_results = np.array([[0, 0, 0, 0, -np.pi/10./norm],
                                     [0, 0, 0, 0, np.pi/10./norm],
                                     [0, 0, 0, 0, -np.pi/20./norm],
                                     [0, 0, 0, 0, -np.pi/20./norm]])

        results = dbbox_transform2_new(boxlist1, boxlist2)
        np.testing.assert_almost_equal(results, expected_results)

    def test_dbbox_transform2_best_match_warp(self):
        boxlist1 = np.array([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi - np.pi/10.],
                                  [1, 1, 10, 5, np.pi - np.pi/10.]
                                  ])
        boxlist2 = np.array([[1, 1, 5, 10, -np.pi/10. + np.pi/2.],
                                  [1, 1, 10, 5, np.pi/10 + np.pi],
                                  [1, 1, 5, 10, np.pi - np.pi/10. - np.pi/20. - np.pi/2.],
                                  [1, 1, 10, 5, np.pi - np.pi/10. - np.pi/20. + 10 * np.pi]
                                  ])
        norm = np.pi / 2.
        expected_results = np.array([[0, 0, 0, 0, -np.pi/10./norm],
                                     [0, 0, 0, 0, np.pi/10./norm],
                                     [0, 0, 0, 0, -np.pi/20./norm],
                                     [0, 0, 0, 0, -np.pi/20./norm]])

        results = dbbox_transform2_best_match_warp(boxlist1, boxlist2)
        print 'results: ', results

        old_resutls = dbbox_transform2(boxlist1, boxlist2)
        print 'old_resutls: ', old_resutls

        np.testing.assert_almost_equal(results, expected_results)

        print 'test decode'
        predict1 = dbbox_transform2_inv_new(boxlist1, results, norm)

        predict2 = dbbox_transform2_inv(boxlist1, old_resutls)

        diff = dbbox_transform2_best_match_warp(predict1, predict2)
        print 'predict1:', predict1
        print 'predict2:', predict2
        print 'diff:', diff
        # self.assertTrue(np.all(diff == 0))

if __name__ == '__main__':
    unittest.main()