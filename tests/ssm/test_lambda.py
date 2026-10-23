r"""Tests for the energy fluctuation variable $\lambda$ and its kinetic energy term."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm import const
from pttools.ssm.ssm import A2_e_conserving, LambdaCorrection, lam, lam_kinetic

#: $v_\text{wall}$ and $\alpha_n$ of the bubbles: the runs of Correia et al. (2025) and weak transitions
PARAMS: tuple[tuple[float, float], ...] = ((0.92, 0.67), (0.44, 0.5), (0.92, 0.0046), (0.44, 0.0046))


class LambdaTest(unittest.TestCase):
    r"""Tests for $\lambda$ with the fluid shells of strong and weak phase transitions."""

    bubbles: tp.ClassVar[list[Bubble]]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.bubbles = [
            Bubble.cached(BagModel.cached(alpha_n_min=min(0.001, alpha_n / 2)), v_wall=v_wall, alpha_n=alpha_n)
            for v_wall, alpha_n in PARAMS
        ]
        for bubble in cls.bubbles:
            if not bubble.solved:
                bubble.solve()

    @staticmethod
    def _integrate(bubble: Bubble, f: np.ndarray) -> float:
        r"""$\int dV f$ over the self-similar coordinate $\xi$.

        The integral is over $\xi^3$, as the interior of the bubble is sampled sparsely.
        """
        return float(4 * np.pi / 3 * np.trapezoid(f, bubble.xi**3))

    def _lam(self, bubble: Bubble, correction: LambdaCorrection) -> np.ndarray:
        return lam(v=bubble.v, w=bubble.w, e=bubble.model.e(bubble.w, bubble.phase), correction=correction)

    def test_lam_kinetic(self) -> None:
        r"""The kinetic terms are $w \gamma^2 v^2 / \bar{w}$, $w v^2 / \bar{w}$ and 0."""
        v = np.array([0., 0.3, 0.6])
        w = np.array([1., 2., 3.])
        np.testing.assert_allclose(
            lam_kinetic(v=v, w=w, w_bar=2., correction=LambdaCorrection.EXACT), w * v**2 / (1 - v**2) / 2, rtol=1e-15)
        np.testing.assert_allclose(
            lam_kinetic(v=v, w=w, w_bar=2., correction=LambdaCorrection.LEGACY), w * v**2 / 2, rtol=1e-15)
        np.testing.assert_array_equal(lam_kinetic(v=v, w=w, w_bar=2., correction=LambdaCorrection.NONE), 0)

    @staticmethod
    def test_lam() -> None:
        r"""$\lambda = (e - \bar{e}) / \bar{w}$ plus the kinetic term.

        By default, $\bar{e} = e_n$ and $\bar{w} = w_n$.
        """
        v = np.array([0.3, 0.1, 0.])
        w = np.array([3., 1., 2.])
        e = np.array([2., 0.5, 1.])
        np.testing.assert_allclose(
            lam(v=v, w=w, e=e, correction=LambdaCorrection.EXACT),
            (e - 1.) / 2. + w * v**2 / (1 - v**2) / 2., rtol=1e-15)
        np.testing.assert_allclose(
            lam(v=v, w=w, e=e, e_bar=0.4, w_bar=1.2, correction=LambdaCorrection.LEGACY),
            (e - 0.4) / 1.2 + w * v**2 / 1.2, rtol=1e-15)

    def test_A2_w_bar(self) -> None:
        r"""$\bar{w}$ scales the $\lambda$ part of $\lvert A(z) \rvert^2$ by $(w_n / \bar{w})^2$."""
        z = np.logspace(-1, 2, 31)
        for (v_wall, alpha_n), bubble in zip(PARAMS[:2], self.bubbles[:2], strict=True):
            with self.subTest(v_wall=v_wall, alpha_n=alpha_n):
                kwargs = {
                    "v": bubble.v, "w": bubble.w, "xi": bubble.xi, "e": bubble.model.e(bubble.w, bubble.phase), "z": z,
                    "v_wall": bubble.v_wall, "v_sh": bubble.v_sh, "cs": np.sqrt(const.CS0_2)}
                _, fp2_n, lam2_n = A2_e_conserving(**kwargs, w_bar=bubble.wn)
                _, fp2, lam2 = A2_e_conserving(**kwargs, w_bar=bubble.w_bar)
                np.testing.assert_array_equal(fp2, fp2_n)
                np.testing.assert_allclose(lam2, (bubble.wn / bubble.w_bar)**2 * lam2_n, rtol=1e-9)
                # By default, w_bar is computed from the profile.
                np.testing.assert_allclose(A2_e_conserving(**kwargs)[2], lam2, rtol=1e-12)

    def test_energy_conservation(self) -> None:
        r"""$\int dV \lambda$ without the kinetic term is that with the exact kinetic term minus the kinetic energy."""
        for (v_wall, alpha_n), bubble in zip(PARAMS, self.bubbles, strict=True):
            with self.subTest(v_wall=v_wall, alpha_n=alpha_n):
                v, w = bubble.v, bubble.w
                kinetic = self._integrate(bubble, w * v**2 / (1 - v**2) / w[-1])
                exact = self._integrate(bubble, self._lam(bubble, LambdaCorrection.EXACT))
                none = self._integrate(bubble, self._lam(bubble, LambdaCorrection.NONE))
                assert none == pytest.approx(exact - kinetic, rel=1e-12)

    def test_energy_conservation_strong(self) -> None:
        r"""For strong transitions, only the exact kinetic term conserves the energy of the fluid shell.

        Without the kinetic term, $\int dV \lambda$ is minus the kinetic energy.
        With the legacy term $w v^2$, it's 28 % of it for the detonation with $\alpha_n = 0.67$.
        For weak transitions the residual of the discretisation of the profile is comparable to the kinetic energy,
        and therefore they are not tested here.
        """
        for (v_wall, alpha_n), bubble in zip(PARAMS[:2], self.bubbles[:2], strict=True):
            with self.subTest(v_wall=v_wall, alpha_n=alpha_n):
                v, w = bubble.v, bubble.w
                kinetic = self._integrate(bubble, w * v**2 / (1 - v**2) / w[-1])
                exact = self._integrate(bubble, self._lam(bubble, LambdaCorrection.EXACT))
                legacy = self._integrate(bubble, self._lam(bubble, LambdaCorrection.LEGACY))
                assert abs(exact) < 0.01 * kinetic
                assert abs(exact) < abs(legacy)
        bubble = self.bubbles[0]
        v, w = bubble.v, bubble.w
        kinetic = self._integrate(bubble, w * v**2 / (1 - v**2) / w[-1])
        assert abs(self._integrate(bubble, self._lam(bubble, LambdaCorrection.LEGACY))) > 0.2 * kinetic

    def test_A2_low_z(self) -> None:
        r"""$\lvert A(z) \rvert^2 \to 0$ as $z \to 0$ only with the kinetic term of $\lambda$."""
        z = np.logspace(-2, 0, 41)
        for (v_wall, alpha_n), bubble in zip(PARAMS[:2], self.bubbles[:2], strict=True):
            with self.subTest(v_wall=v_wall, alpha_n=alpha_n):
                A2 = {
                    correction: A2_e_conserving(
                        v=bubble.v, w=bubble.w, xi=bubble.xi, e=bubble.model.e(bubble.w, bubble.phase), z=z,
                        v_wall=bubble.v_wall, v_sh=bubble.v_sh, cs=np.sqrt(const.CS0_2),
                        lambda_correction=correction)[0]
                    for correction in LambdaCorrection
                }
                # Without the kinetic term, |A|^2 approaches a non-zero constant.
                assert A2[LambdaCorrection.NONE][0] == pytest.approx(A2[LambdaCorrection.NONE][20], rel=0.1)
                # With it, |A|^2 decreases by at least a factor of 10 per decade of z.
                assert A2[LambdaCorrection.EXACT][0] < A2[LambdaCorrection.EXACT][20] / 10
                assert A2[LambdaCorrection.EXACT][0] < 1e-2 * A2[LambdaCorrection.NONE][0]


if __name__ == "__main__":
    unittest.main()
