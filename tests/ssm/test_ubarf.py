r"""Tests for the root-mean-square fluid velocity $\bar{U}_f$."""

from math import sqrt
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.math import assert_allclose
from pttools.models.bag import BagModel
from pttools.speedup.options import NUMBA_DISABLE_JIT
from pttools.ssm import LambdaCorrection, NucType, SSMSpectrum
from pttools.ssm.const import QT_INTEGRALS_MAX, QT_INTEGRALS_MIN, QT_INTEGRALS_POINTS_PER_DECADE
from pttools.ssm.low_k.integration import Iv, Iv_resampled
from pttools.ssm.ssm import qT_integrals
import pttools.type_hints as th

#: :gw_pt_ssm:`\ ` table 1
TABLE1: th.FloatArr2D = np.array([
    [0.46] * 5 + [5.] * 5,
    [0.92, 0.80, 0.68, 0.56, 0.44, 0.92, 0.80, 0.73, 0.56, 0.44],
    [4.5, 5.8, 9.0, 16.1, 8.5, 46.1, 59.7, 77.5, 102.8, 80.8],
    [4.5, 5.8, 9.0, 16.2, 8.5, 46.1, 59.8, 77.7, 103.0, 80.9],
    [5.2, 6.5, 9.7, 16.2, 7.6, 54.7, 68.4, 86.4, 100.3, 71.5]
])
#: $\alpha_n$ of :gw_pt_ssm:`\ ` table 1
ALPHA_N = TABLE1[0] * 0.01
#: $v_\text{wall}$ of :gw_pt_ssm:`\ ` table 1
V_WALL = TABLE1[1]
#: $\bar{U}_{f,3}^\text{sim}$ of :gw_pt_ssm:`\ ` table 1
UBARF_SIM = TABLE1[2] * 0.001
#: $\bar{U}_{f,3}^\text{exp}$ of :gw_pt_ssm:`\ ` table 1
UBARF_EXP = TABLE1[3] * 0.001
#: $\bar{U}_{f,3}^\text{1d}$ of :gw_pt_ssm:`\ ` table 1
UBARF_1D = TABLE1[4] * 0.001

# Print the table in the same orientation as in the article.
# print(TABLE1.T)


class UbarfTest(unittest.TestCase):
    r"""Test $\bar{U}_f$ against :gw_pt_ssm:`\ ` table 1."""

    model: tp.ClassVar[BagModel]
    bubbles: tp.ClassVar[list[Bubble]]
    spectra: tp.ClassVar[list[SSMSpectrum]]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.model = BagModel.cached(alpha_n_min=ALPHA_N[0])
        cls.bubbles = [
            Bubble.cached(cls.model, v_wall=v_wall, alpha_n=alpha_n)
            for v_wall, alpha_n in zip(V_WALL, ALPHA_N, strict=False)
        ]
        spectrum_kwargs: dict[str, tp.Any] = {}
        if NUMBA_DISABLE_JIT:
            # Without Numba, the default accuracy would take more than 10 minutes.
            # These settings affect only the GW spectra, and therefore give the same Ubarf as the defaults.
            spectrum_kwargs = {"n_z_lookup": 2000, "nx_P_tilde_gw": 100}
        cls.spectra = [SSMSpectrum(bubble, compute=False, **spectrum_kwargs) for bubble in cls.bubbles]
        for spectrum in cls.spectra:
            # The reference values were computed with the legacy kinetic energy term of lambda.
            spectrum.compute(lambda_correction=LambdaCorrection.LEGACY)

    def test_ubarf_1d(self) -> None:
        r"""Test $\bar{U}_f$ of the bubbles against the 1D values of :gw_pt_ssm:`\ ` table 1."""
        assert_allclose(
            [bubble.ubarf for bubble in self.bubbles],
            UBARF_1D,
            rtol=0.023
        )

    def test_ubarf_exponential(self) -> None:
        r"""Test $\bar{U}_f$ for exponential nucleation against :gw_pt_ssm:`\ ` table 1."""
        ubarf = [sqrt(spectrum.ubarf2_custom_nucleation(nuc_type=NucType.EXPONENTIAL)) for spectrum in self.spectra]
        # The hybrid with alpha_n=0.05, v_wall=0.73 deviated by 75 % before the thin shell of hybrids was preserved
        # in the sine transform of lambda, and by 9.5 % before the exact sine transforms were used for large z.
        # The values of the article have only two significant digits.
        assert_allclose(
            ubarf,
            UBARF_EXP,
            rtol=0.02
        )

    def test_ubarf_simultaneous(self) -> None:
        r"""Test $\bar{U}_f$ for simultaneous nucleation against :gw_pt_ssm:`\ ` table 1."""
        ubarf = [sqrt(spectrum.ubarf2_custom_nucleation(nuc_type=NucType.SIMULTANEOUS)) for spectrum in self.spectra]
        # See test_ubarf_exponential
        assert_allclose(
            ubarf,
            UBARF_SIM,
            rtol=0.02
        )

    def test_w_bar(self) -> None:
        r"""$\bar{w}$ and $w_n$ are not the same, but they should be somewhat close."""
        assert_allclose(
            [bubble.w_bar for bubble in self.bubbles],
            [bubble.wn for bubble in self.bubbles],
            rtol=0.05
        )


class IntegralsTest(unittest.TestCase):
    r"""Test that $\bar{U}_f^2$ and $\mathcal{I}_v$ of the spectra do not depend on their $y$ array.

    See :py:func:`pttools.ssm.ssm.qT_integrals`.
    """

    bubble: tp.ClassVar[Bubble]
    #: $y$ arrays with different ranges and densities
    Y_ARRAYS: tp.ClassVar[tuple[th.FloatArr1D, ...]] = (np.logspace(-3, 0, 30), np.logspace(-1, 3, 300))
    #: Accuracy settings that do not affect $\bar{U}_f^2$
    SPECTRUM_KWARGS: tp.ClassVar[dict[str, tp.Any]] = {"n_z_lookup": 1000, "nx_P_tilde_gw": 100, "low_k": False}

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.bubble = Bubble.cached(BagModel.cached(alpha_n_min=0.01), v_wall=0.8, alpha_n=0.2)

    def spectrum(self, y: th.FloatArr1D, **kwargs: tp.Any) -> SSMSpectrum:
        """A spectrum of the bubble with the given $y$ array."""
        return SSMSpectrum(self.bubble, y=y, r_star=0.01, **(self.SPECTRUM_KWARGS | kwargs))

    def test_independent_of_y(self) -> None:
        r"""$\bar{U}_f^2$ and $\mathcal{I}_v$ are the same for $y$ arrays with different ranges and densities."""
        spectra = [self.spectrum(y) for y in self.Y_ARRAYS]
        assert_allclose(spectra[0].ubarf2, spectra[1].ubarf2, rtol=1e-12)
        assert_allclose(spectra[0].Iv, spectra[1].Iv, rtol=1e-12)
        for spectrum in spectra:
            assert_allclose(spectrum.ubarf2_custom_nucleation(), spectrum.ubarf2, rtol=1e-12)

    def test_legacy(self) -> None:
        r"""With ``qT_integrals_max=0``, $\bar{U}_f^2$ is integrated over the lookup array of $y \tilde{T}$.

        The integral is then cut off at $y_\text{max} \tilde{T}_\text{max}$,
        which underestimates $\bar{U}_f^2$ for the short $y$ array by more than a factor of 2.
        """
        y = self.Y_ARRAYS[0]
        legacy = self.spectrum(y, qT_integrals_max=0)
        np.testing.assert_array_equal(legacy.qT_integrals, legacy.qT_lookup)
        assert legacy.ubarf2 < 0.5 * self.spectrum(y).ubarf2
        assert legacy.Iv == Iv_resampled(x=legacy.z_lookup, P_tilde_v=legacy.spec_den_v_lookup)

    def test_converged(self) -> None:
        r"""Increasing the upper limit of the integral by a factor of 10 changes $\bar{U}_f^2$ only a little.

        The difference is mostly the sampling error of the oscillating integrand,
        see :py:data:`pttools.ssm.const.QT_INTEGRALS_POINTS_PER_DECADE`.
        """
        y = self.Y_ARRAYS[0]
        assert_allclose(
            self.spectrum(y, qT_integrals_max=10 * QT_INTEGRALS_MAX).ubarf2, self.spectrum(y).ubarf2, rtol=3e-3)

    def test_iv(self) -> None:
        r"""$\mathcal{I}_v$ agrees with that integrated over a wide lookup array of $\tilde{P}_v$.

        The lookup array of the spectrum with $y$ from $10^{-3}$ to $10^4$ covers $x$ from about $4 \cdot 10^{-4}$
        to $1.4 \cdot 10^4$, i.e. practically the whole support of $\tilde{P}_v$.
        """
        wide = self.spectrum(np.logspace(-3, 4, 700), n_z_lookup=14000)
        assert_allclose(wide.Iv, Iv(x=wide.z_lookup, P_tilde_v=wide.spec_den_v_lookup), rtol=1e-3)
        base_res = wide.base_res
        assert base_res is not None
        assert_allclose(wide.Iv, Iv(x=base_res.x_Iv, P_tilde_v=base_res.spec_den_v_Iv), rtol=1e-12)

    @staticmethod
    def test_qT_integrals() -> None:
        r"""The array covers $[q\tilde{T}_\text{min}, \max(q\tilde{T}_\text{max}, \max(q\tilde{T}_\text{lookup}))]$."""
        for qT_lookup in (np.logspace(-5, 1, 100), np.logspace(-1, 7, 100)):
            qT = qT_integrals(qT_lookup)
            assert qT[0] == pytest.approx(QT_INTEGRALS_MIN, rel=1e-12)
            assert qT[-1] == pytest.approx(max(QT_INTEGRALS_MAX, qT_lookup[-1]), rel=1e-12)
            assert np.log10(qT[1] / qT[0]) <= 1 / QT_INTEGRALS_POINTS_PER_DECADE * (1 + 1e-9)
        qT_lookup = np.logspace(-1, 1, 10)
        assert qT_integrals(qT_lookup, qT_max=0) is qT_lookup


if __name__ == "__main__":
    unittest.main()
