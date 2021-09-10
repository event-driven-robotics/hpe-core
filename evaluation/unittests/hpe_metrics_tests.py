
import numpy as np
import unittest

from numpy.testing import assert_allclose

from ..utils.hpe_metrics import compute_pck


class TestHepMetrics(unittest.TestCase):

    POSES_2D_GT = np.array([])
    POSES_2D_PRED = np.array([])
    PCK_2D = np.array([])

    POSES_3D_GT = np.array([])
    POSES_3D_PRED = np.array([])
    PCK_3D = np.array([])

    RTOL = 1e-5

    def test_compute_pck(self):

        pck = compute_pck(predicted_joints=self.POSES_2D_PRED, gt_joints=self.POSES_2D_GT, threshold=.1, head_size=20)
        assert_allclose(actual=pck, desired=self.PCK_2D, rtol=self.RTOL, err_msg='Failed to compute 2D PCK')

        pck = compute_pck(predicted_joints=self.POSES_3D_PRED, gt_joints=self.POSES_3D_GT, threshold=.1, head_size=20)
        assert_allclose(actual=pck, desired=self.PCK_3D, rtol=self.RTOL, err_msg='Failed to compute 3D PCK')


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
