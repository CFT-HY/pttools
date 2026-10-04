"""Tests for the fluid profile curves of the plots."""

import typing as tp
import unittest

import numpy as np

from pttools.analysis import fluid
from pttools.bubble import v_max_behind
from pttools.bubble.const import CS0
import pttools.type_hints as th


class CurvesSymmetricTest(unittest.TestCase):
    """Tests for the fluid profile curves in the symmetric phase."""

    N_XI: int = 50
    data: th.FloatArr3D
    data_cut: th.FloatArr3D

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.data = fluid.curves_symmetric(n_xi=cls.N_XI)
        cls.data_cut = fluid.curves_symmetric(csb=CS0, n_xi=cls.N_XI)

    def test_shape(self) -> None:
        """Test that the curves have the documented shape and that the default curves can be integrated."""
        self.assertEqual(self.data.shape, (3, fluid.DEFAULT_CURVES_SYMMETRIC_V.size, 2*self.N_XI))
        self.assertFalse(np.any(np.isnan(self.data)))

    def test_start_on_v_xi_line(self) -> None:
        r"""Test that both the backwards and forwards curves start from the $v = \xi$ line."""
        v0 = fluid.DEFAULT_CURVES_SYMMETRIC_V
        for i_start in (0, self.N_XI):
            np.testing.assert_allclose(self.data[0, :, i_start], v0, rtol=1e-12)
            np.testing.assert_allclose(self.data[2, :, i_start], v0, rtol=1e-12)

    def test_csb_removes_below_mu(self) -> None:
        r"""Test that csb removes the part of the backwards curves below the $\mu(\xi, v) = c_{s,b}$ curve."""
        v_b = self.data[0, :, :self.N_XI]
        xi_b = self.data[2, :, :self.N_XI]
        below = v_b < v_max_behind(xi_b, CS0)
        # Ensure that the test is meaningful
        self.assertTrue(np.any(below))
        self.assertTrue(np.any(~below))

        v_b_cut = self.data_cut[0, :, :self.N_XI]
        self.assertTrue(np.all(np.isnan(v_b_cut[below])))
        np.testing.assert_array_equal(v_b_cut[~below], v_b[~below])

    def test_csb_keeps_other_data(self) -> None:
        r"""Test that csb does not affect $w$, $\xi$ or the forwards curves."""
        np.testing.assert_array_equal(self.data_cut[1:], self.data[1:])
        np.testing.assert_array_equal(self.data_cut[0, :, self.N_XI:], self.data[0, :, self.N_XI:])
