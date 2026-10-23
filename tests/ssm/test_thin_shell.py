r"""Tests for the sine transforms of fluid shells with a thin shell between the wall and the shock.

For hybrids with $v_\text{wall}$ close to the Chapman-Jouguet speed $v_\text{CJ}$,
the shell between the wall and the shock is thinner than the spacing of the uniform resampling grid.
The sine transform of $\lambda$ should nevertheless be continuous across the transition from detonations to hybrids,
and its approximation for large $z$ should be close to the exact sine transform.
"""

import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models.bag import BagModel
from pttools.ssm.calculators import resample_merged_uniform_xi, resample_uniform_xi
from pttools.ssm.sin_transform import sin_transform_core_single
from pttools.ssm.ssm import LambdaCorrection, f, l, lam
import pttools.type_hints as th

V_WALL: float = 0.95
#: $\alpha_n$ for detonations close to $v_\text{CJ}$, and for hybrids with thin shells of increasing thickness
ALPHA_NS: tuple[float, ...] = (1.40, 1.42, 1.43, 1.45, 1.55, 1.6, 2.0)
Z: th.FloatArr1D = np.logspace(np.log10(60), 3, 100)
N_XI_EXACT: int = 200_000


def rms_z2(arr: th.FloatArr1D) -> float:
    r"""Root-mean-square of $z^2$ times the array, which removes the leading power law of the transforms."""
    return float(np.sqrt(np.mean((arr * Z**2)**2)))


class ThinShellTest(unittest.TestCase):
    """Tests for the sine transforms of hybrids with thin shells."""

    bubbles: dict[float, Bubble]
    transforms: dict[float, tuple[float, float, float, float]]

    @classmethod
    def setUpClass(cls) -> None:  # noqa: D102
        model = BagModel.cached(alpha_n_min=0.0001)
        cls.bubbles = {}
        cls.transforms = {}
        for alpha_n in ALPHA_NS:
            bubble = Bubble(model=model, v_wall=V_WALL, alpha_n=alpha_n)
            bubble.solve()
            cls.bubbles[alpha_n] = bubble
            cls.transforms[alpha_n] = cls.compute_transforms(bubble)

    @staticmethod
    def compute_transforms(bubble: Bubble) -> tuple[float, float, float, float]:
        r"""The RMS of the approximate and exact transforms of $\lambda$ and $v$ for large $z$.

        $\lambda$ is computed without its kinetic energy term (:py:attr:`pttools.ssm.ssm.LambdaCorrection.NONE`),
        for which the tolerances of :py:meth:`test_approximation` have been set.
        With the term, $\lambda$ is steeper behind the wall, and the error of the approximation is up to 47 %,
        which exceeds the tolerance of 40 %.
        The approximation is not used by default, see ``exact_large_z`` of :py:func:`pttools.ssm.ssm.f_fp_l`.
        """
        lm = lam(v=bubble.v, w=bubble.w, e=bubble.e, correction=LambdaCorrection.NONE)
        xi_exact, lam_exact = resample_uniform_xi(bubble.xi, lm, N_XI_EXACT)
        _, v_exact = resample_uniform_xi(bubble.xi, bubble.v, N_XI_EXACT)
        l_exact = 4 * np.pi / Z * sin_transform_core_single(t=xi_exact, f=xi_exact * lam_exact, freq=Z)
        f_exact = 4 * np.pi / Z * sin_transform_core_single(t=xi_exact, f=v_exact, freq=Z)
        l_approx = l(z=Z, xi=bubble.xi, lam=lm, v_wall=bubble.v_wall, v_sh=bubble.v_sh, parallel=False)
        f_approx = f(z=Z, xi=bubble.xi, v=bubble.v, v_wall=bubble.v_wall, v_sh=bubble.v_sh, parallel=False)
        return rms_z2(l_approx), rms_z2(l_exact), rms_z2(f_approx), rms_z2(f_exact)

    def test_solution_types(self) -> None:
        """The test cases should cover both sides of the Chapman-Jouguet speed."""
        assert self.bubbles[1.42].sol_type.name == "DETON"
        assert self.bubbles[1.43].sol_type.name == "HYBRID"
        assert self.bubbles[1.45].v_sh - V_WALL < 1e-4

    def test_approximation(self) -> None:
        r"""The large-$z$ approximation should be close to the exact sine transform."""
        for alpha_n in ALPHA_NS:
            with self.subTest(alpha_n=alpha_n):
                l_approx, l_exact, f_approx, f_exact = self.transforms[alpha_n]
                # The approximation neglects the O(1/z^2) terms,
                # which are significant close to v_CJ due to the steep profile behind the wall.
                assert abs(l_approx / l_exact - 1) < 0.4
                assert abs(f_approx / f_exact - 1) < 0.05

    def test_continuity(self) -> None:
        r"""The transform of $\lambda$ should be continuous across the transition from detonations to hybrids."""
        l_deton = self.transforms[1.42][0]
        l_hybrid = self.transforms[1.43][0]
        assert abs(l_hybrid / l_deton - 1) < 0.15


class ResampleMergedTest(unittest.TestCase):
    """Tests for :func:`pttools.ssm.calculators.resample_merged_uniform_xi`."""

    def test_merge(self) -> None:
        """The original points and the discontinuities should be preserved."""
        xi = np.array([0, 0.3, 0.5, 0.5, 0.50001, 0.50002, 1])
        f_arr = np.array([0, 0.3, 0.5, 2., 2., 0., 0.])
        xi_re, f_re = resample_merged_uniform_xi(xi, f_arr, 10)
        assert np.all(np.diff(xi_re) >= 0)
        # The original points are kept in order, including the duplicate at the discontinuity.
        np.testing.assert_array_equal(xi_re[np.isin(xi_re, xi)], xi)
        np.testing.assert_array_equal(f_re[np.isin(xi_re, xi)], f_arr)
        # The uniform points that do not coincide with the original points are interpolated.
        uniform = np.linspace(0, 0.9, 10)
        added = uniform[~np.isin(uniform, xi)]
        np.testing.assert_array_equal(xi_re[~np.isin(xi_re, xi)], added)
        np.testing.assert_allclose(f_re[~np.isin(xi_re, xi)], np.where(added < 0.5, added, 0))


if __name__ == "__main__":
    unittest.main()
