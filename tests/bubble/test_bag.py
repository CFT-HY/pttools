import unittest

import numpy as np

from pttools import bubble
from pttools.bubble.boundary import Phase
from pttools.models.bag import BagModel
from tests import utils


class TestBag(unittest.TestCase):
    """Test the functions of the bag model equation of state

    TODO: This will be phased out when the support for custom models is implemented
    """
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

    # @staticmethod
    # def junction_bag(
    #         model: BagModel,
    #         v1: np.ndarray, w1: np.ndarray,
    #         phase1: Phase, phase2: Phase,
    #         greater_branch: bool,
    #         atol=1.3e-14):
    #     zeros = np.zeros_like(v1)
    #     v2, w2 = bubble.junction_bag(v1, w1, model.V_b, model.V_s, greater_branch=greater_branch)
    #     data1 = bubble.junction_condition_deviation1(v1, w1, v2, w2)
    #     utils.assert_allclose(data1, zeros, atol=atol)
    #
    #     data2 = bubble.junction_condition_deviation2(
    #         v1, w1, model.p(w1, phase1),
    #         v2, w2, model.p(w2, phase2)
    #     )
    #     utils.assert_allclose(data2, zeros, atol=atol)

    # def test_adiabatic_index(self):
    #     pass

    def test_e(self):
        ref_data = np.array([1.175, 1.2125, 1.25, 1.2875, 1.325])
        data = bubble.get_e(self.w_arr, self.phase, self.theta_s, self.theta_b)
        utils.assert_allclose(data, ref_data)

    # def test_junction_bag_lesser(self):
    #     model = BagModel(V_s=1, a_s=1.1, a_b=1)
    #     v1 = np.linspace(0.1, 0.9, 10)
    #     w1 = np.linspace(0.9, 1.1, 10)
    #     self.junction_bag(model, v1, w1, Phase.BROKEN, Phase.SYMMETRIC, greater_branch=False)
    #
    # def test_junction_bag_greater(self):
    #     model = BagModel(V_s=1, a_s=1.1, a_b=1)
    #     v1 = np.linspace(0.4, 0.9, 10)
    #     w1 = np.linspace(1.5, 2, 10)
    #     self.junction_bag(model, v1, w1, Phase.BROKEN, Phase.SYMMETRIC, greater_branch=True)

    def test_p(self):
        ref_data = np.array([-0.275, -0.2625, -0.25, -0.2375, -0.225])
        data = bubble.get_p(self.w_arr, self.phase, self.theta_s, self.theta_b)
        utils.assert_allclose(data, ref_data)

    # def test_phase_scalar(self):
    #     pass

    def test_theta_bag_scalar(self):
        self.assertEqual(bubble.theta_bag(self.w, 1, self.alpha_n), 0)
        self.assertAlmostEqual(bubble.theta_bag(self.w, self.phase, self.alpha_n), 0.3375)

    def test_theta_bag_arr(self):
        self.assertEqual(bubble.theta_bag(self.w_arr, 1, self.alpha_n), 0)
        self.assertAlmostEqual(bubble.theta_bag(self.w_arr, self.phase, self.alpha_n), 0.4125)

    def test_w(self):
        ref_data = np.array([0.9, 0.95, 1, 1.05, 1.1])
        e = bubble.get_e(self.w_arr, self.phase, self.theta_s, self.theta_b)
        data = bubble.get_w(e, self.phase, self.theta_s, self.theta_b)
        utils.assert_allclose(data, ref_data)


if __name__ == "__main__":
    unittest.main()
