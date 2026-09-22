r"""Tests for the kinetic energy suppression factors of the Sound Shell Model."""

import unittest

import numpy as np

from pttools.ssm.suppression import suppression as sup
import pttools.type_hints as th
from pttools.utils.assertions import assert_allclose


class AlphaNMaxTest(unittest.TestCase):
    r"""Tests for $\alpha_{n,\max}({v}_\text{wall})$."""

    #: The $({v}_\text{wall}, \alpha_n)$ points of the suppression dataset,
    #: from which the piecewise linear fit of :py:func:`pttools.ssm.suppression.alpha_n_max` is constructed.
    DATA_POINTS: tuple[tuple[float, float], ...] = ((0.24, 0.34), (0.44, 0.50), (0.56, 0.67))
    #: ${v}_\text{wall}$ at which the fit changes from the first line segment to the second
    V_WALL_KINK: float = 0.44

    def test_data_points(self):
        """The fit goes through the data points from which it was constructed."""
        v_walls = np.array([point[0] for point in self.DATA_POINTS])
        alpha_ns = np.array([point[1] for point in self.DATA_POINTS])
        assert_allclose(sup.alpha_n_max(v_walls), alpha_ns)

    def test_scalar_matches_array(self):
        """A scalar argument gives the same result as the corresponding element of an array argument.

        This covers both line segments, as the fit is piecewise linear.
        """
        v_walls = np.linspace(0.2, 0.9, 15)
        alpha_ns = sup.alpha_n_max(v_walls)
        for v_wall, alpha_n in zip(v_walls, alpha_ns, strict=True):
            with self.subTest(v_wall=v_wall):
                self.assertAlmostEqual(sup.alpha_n_max(float(v_wall)), alpha_n)

    def test_continuous_at_kink(self):
        """The two line segments meet at the point that they have in common."""
        alpha_n_kink = self.DATA_POINTS[1][1]
        self.assertAlmostEqual(sup.alpha_n_max(self.V_WALL_KINK), alpha_n_kink)
        self.assertAlmostEqual(sup.alpha_n_max(self.V_WALL_KINK - 1e-9), alpha_n_kink)
        self.assertAlmostEqual(sup.alpha_n_max(self.V_WALL_KINK + 1e-9), alpha_n_kink)

    def test_increasing(self):
        r"""$\alpha_{n,\max}$ increases with ${v}_\text{wall}$."""
        alpha_ns = sup.alpha_n_max(np.linspace(0.2, 0.9, 50))
        self.assertTrue(np.all(np.diff(alpha_ns) > 0))

    def test_approx_scalar_matches_array(self):
        """The approximation gives the same result for a scalar and for an array argument."""
        v_walls = np.linspace(0.1, 0.9, 9)
        alpha_ns = sup.alpha_n_max_approx(v_walls)
        for v_wall, alpha_n in zip(v_walls, alpha_ns, strict=True):
            with self.subTest(v_wall=v_wall):
                self.assertAlmostEqual(sup.alpha_n_max_approx(float(v_wall)), alpha_n)

    def test_approx_limits(self):
        r"""$\alpha_{n,\max,\text{approx}} = \frac{1}{3}$ for ${v}_\text{wall}=0$ and 1 for ${v}_\text{wall}=c_s$.

        $$\alpha_{n,\max,\text{approx}}
        = \frac{1}{3} \frac{1 + 3 {v}_\text{wall}^2}{1 - {v}_\text{wall}^2}$$
        """
        self.assertAlmostEqual(sup.alpha_n_max_approx(0.), 1/3)
        self.assertAlmostEqual(sup.alpha_n_max_approx(1/np.sqrt(3)), 1.)


class SuppressionTest(unittest.TestCase):
    """Tests for :py:class:`pttools.ssm.suppression.Suppression`."""

    #: The dataset is given explicitly instead of using DEFAULT_SUPPRESSION,
    #: so that the reference values stay valid even if the default is changed.
    SUPPRESSION: sup.Suppression = sup.NO_HYBRIDS_EXT
    V_WALLS: th.FloatArr1D = np.array([0.4, 0.5, 0.6])
    ALPHA_NS: th.FloatArr1D = np.array([0.05, 0.1])
    #: Reference values, computed with PTtools 0.10.0
    GRID_REF: th.FloatArr2D = np.array([
        [0.2175166, 0.60930056, 0.87699788],
        [0.16514226, 0.45835392, 0.6073551]
    ])
    #: A point outside the convex hull of the suppression points
    OUTSIDE_V_WALL: float = 0.95
    OUTSIDE_ALPHA_N: float = 0.9
    #: Reference value for the nearest-neighbour extrapolation, computed with PTtools 0.10.0
    OUTSIDE_REF: float = 0.378071924042103

    def test_grid(self):
        """Interpolating on a grid of points gives the reference values."""
        grid = self.SUPPRESSION.suppression(
            v_wall=self.V_WALLS, alpha_n=self.ALPHA_NS, method=sup.SuppressionMethod.EXT_CONSTANT)
        assert_allclose(grid, self.GRID_REF, rtol=1e-7)

    def test_grid_indexing(self):
        r"""The grid is indexed as ``[alpha_n, v_wall]``, and its values are those of the scalar calls."""
        grid = self.SUPPRESSION.suppression(
            v_wall=self.V_WALLS, alpha_n=self.ALPHA_NS, method=sup.SuppressionMethod.EXT_CONSTANT)
        self.assertEqual(grid.shape, (self.ALPHA_NS.size, self.V_WALLS.size))
        for i, alpha_n in enumerate(self.ALPHA_NS):
            for j, v_wall in enumerate(self.V_WALLS):
                with self.subTest(v_wall=v_wall, alpha_n=alpha_n):
                    scalar = self.SUPPRESSION.suppression(
                        v_wall=float(v_wall), alpha_n=float(alpha_n),
                        method=sup.SuppressionMethod.EXT_CONSTANT)
                    self.assertAlmostEqual(scalar, grid[i, j])

    def test_no_ext_outside_hull(self):
        """Without extrapolation the suppression factor is NaN outside the convex hull of the points."""
        with self.assertLogs(sup.logger, level="WARNING"):
            value = self.SUPPRESSION.suppression(
                v_wall=self.OUTSIDE_V_WALL, alpha_n=self.OUTSIDE_ALPHA_N, method=sup.SuppressionMethod.NO_EXT)
        self.assertTrue(np.isnan(value))

    def test_ext_constant_outside_hull(self):
        """With constant extrapolation the nearest suppression value is returned outside the convex hull."""
        value = self.SUPPRESSION.suppression(
            v_wall=self.OUTSIDE_V_WALL, alpha_n=self.OUTSIDE_ALPHA_N, method=sup.SuppressionMethod.EXT_CONSTANT)
        self.assertAlmostEqual(value, self.OUTSIDE_REF)

    def test_none_scalar(self):
        """Disabling the suppression gives a suppression factor of 1 for scalar arguments."""
        value = self.SUPPRESSION.suppression(v_wall=0.5, alpha_n=0.1, method=sup.SuppressionMethod.NONE)
        self.assertEqual(value, 1.)

    def test_none_grid(self):
        """Disabling the suppression gives a grid of ones with the same shape as the interpolated grid."""
        cases: tuple[tuple[th.FloatOrArr, th.FloatOrArr], ...] = (
            (self.V_WALLS, self.ALPHA_NS),
            (self.V_WALLS, float(self.ALPHA_NS[0])),
            (float(self.V_WALLS[0]), self.ALPHA_NS),
        )
        for v_wall, alpha_n in cases:
            with self.subTest(v_wall=v_wall, alpha_n=alpha_n):
                interpolated = self.SUPPRESSION.suppression(
                    v_wall=v_wall, alpha_n=alpha_n, method=sup.SuppressionMethod.EXT_CONSTANT)
                ones = self.SUPPRESSION.suppression(
                    v_wall=v_wall, alpha_n=alpha_n, method=sup.SuppressionMethod.NONE)
                self.assertEqual(np.shape(ones), np.shape(interpolated))
                assert_allclose(ones, np.ones_like(interpolated))

    def test_invalid_method(self):
        """An invalid suppression method raises an error."""
        with self.assertRaises(ValueError):
            # An invalid value is passed on purpose, and therefore the type error is ignored.
            # pyrefly: ignore[bad-argument-type]
            self.SUPPRESSION.suppression(v_wall=0.5, alpha_n=0.1, method="invalid")

    def test_peak(self):
        """The peak is the point with the highest suppression factor."""
        v_wall, alpha_n, suppression = self.SUPPRESSION.peak()
        self.assertEqual(suppression, self.SUPPRESSION.suppressions.max())
        index = np.argmax(self.SUPPRESSION.suppressions)
        self.assertEqual(v_wall, self.SUPPRESSION.v_walls[index])
        self.assertEqual(alpha_n, self.SUPPRESSION.alpha_ns[index])

    def test_invalid_data(self):
        """The input arrays must have the same size."""
        with self.assertRaises(ValueError):
            sup.Suppression(
                v_walls=np.array([0.5, 0.6]),
                alpha_ns=np.array([0.1]),
                suppressions=np.array([0.5, 0.6]),
                name="Invalid"
            )


if __name__ == "__main__":
    unittest.main()
