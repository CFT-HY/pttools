r"""Tests for the sine transform $\hat{f}(z)$ of the Sound Shell Model."""

import unittest

import numpy as np

from pttools.ssm import const
from pttools.ssm.sin_transform import sin_transform
import pttools.type_hints as th
from pttools.utils import assert_allclose

XI: th.FloatArr1D = np.linspace(0, 1, 5001)
F: th.FloatArr1D = np.ones_like(XI)


def sin_transform_exact(z: th.FloatArr1D) -> th.FloatArr1D:
    r"""Sine transform of $f(\xi) = 1$ on $\xi \in [0, 1]$: $\int_0^1 \sin(z \xi) d\xi = \frac{1 - \cos z}{z}$."""
    return (1 - np.cos(z)) / z


class TestSinTransform(unittest.TestCase):
    """Tests for the sine transform."""

    @staticmethod
    def test_below_blend() -> None:
        """Below the blend range the sine transform is computed exactly."""
        z = np.linspace(0.1, const.Z_ST_THRESH - const.DZ_ST_BLEND - 1, 50)
        assert_allclose(sin_transform(z, XI, F), sin_transform_exact(z), atol=1e-6)

    @staticmethod
    def test_blend_range_without_approximation() -> None:
        """If no z is above the threshold, the approximation should not be blended into the results."""
        z = np.linspace(0.1, const.Z_ST_THRESH - 1, 50)
        data = sin_transform(z, XI, F)
        assert_allclose(data, sin_transform_exact(z), atol=1e-6)
        scalar = np.array([sin_transform(float(z_i), XI, F) for z_i in z])
        assert_allclose(data, scalar, rtol=1e-12, atol=1e-15)

    @staticmethod
    def test_across_threshold() -> None:
        """Reference values for z values across the threshold, including the blend range."""
        z = np.linspace(40, 60, 11)
        ref = np.array([
            4.1673229282661220e-02, 3.3332787692024225e-02, 3.5611441164602290e-06, 3.1134083548741136e-02,
            3.4169411315444452e-02, -7.0067943015773349e-04, -2.2365207322994334e-02, -3.3876108015984265e-02,
            -2.6210695049538541e-03, -1.5186549388813459e-02, -3.2540216340252608e-02
        ])
        assert_allclose(sin_transform(z, XI, F), ref, rtol=1e-12, atol=1e-15)
