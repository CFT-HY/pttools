import unittest

import numpy as np

from pttools import bubble
# from tests.test_utils import print_high_prec as php


class TestBag(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        phase = order parameter (int?), test separately with values 0 and 1
        theta_s = some small positive number, 0.5
        w = usually 1 or around it, can be above or below 1
        """
        cls.alpha_n = 0.5
        # For the 1 phase many results would go to zero
        cls.phase = 0
        cls.theta_s = 0.5
        cls.theta_b = 0.1
        cls.w = 0.9
        cls.w_arr = np.linspace(0.9, 1.1, 5)

    def test_adiabatic_index(self):
        pass

    def test_e(self):
        ref_data = [1.175, 1.2125, 1.25, 1.2875, 1.325]
        data = bubble.get_e(self.w_arr, self.phase, self.theta_s, self.theta_b)
        np.testing.assert_allclose(data, ref_data)

    def test_p(self):
        ref_data = [-0.275, -0.2625, -0.25, -0.2375, -0.225]
        data = bubble.get_p(self.w_arr, self.phase, self.theta_s, self.theta_b)
        np.testing.assert_allclose(data, ref_data)

    def test_phase_scalar(self):
        pass

    def test_theta_bag_scalar(self):
        self.assertEqual(bubble.theta_bag(self.w, 1, self.alpha_n), 0)
        self.assertAlmostEqual(bubble.theta_bag(self.w, self.phase, self.alpha_n), 0.3375)

    def test_theta_bag_arr(self):
        self.assertEqual(bubble.theta_bag(self.w_arr, 1, self.alpha_n), 0)
        self.assertAlmostEqual(bubble.theta_bag(self.w_arr, self.phase, self.alpha_n), 0.4125)

    def test_w(self):
        ref_data = [0.9, 0.95, 1, 1.05, 1.1]
        e = bubble.get_e(self.w_arr, self.phase, self.theta_s, self.theta_b)
        data = bubble.get_w(e, self.phase, self.theta_s, self.theta_b)
        np.testing.assert_allclose(data, ref_data)


if __name__ == "__main__":
    unittest.main()
