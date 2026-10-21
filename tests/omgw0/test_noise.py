r"""Tests for the signal-to-noise ratio of $\Omega_{\text{gw},0}$."""

import unittest

import numpy as np

from pttools.omgw0.noise import omega_noise_h2, signal_to_noise_ratio

#: Frequencies of the test signal (Hz)
F: np.ndarray = np.logspace(-5, 0, 101)
#: Test signal, a broken power law peaking near 1 mHz
SIGNAL: np.ndarray = 1e-10 * (F / 1e-3)**3 / (1 + (F / 1e-3)**4)
#: Frequencies of the separately given noise (Hz)
F_NOISE: np.ndarray = np.logspace(-4, -1, 51)


class SignalToNoiseRatioTest(unittest.TestCase):
    """Tests for :py:func:`pttools.omgw0.noise.signal_to_noise_ratio`."""

    @staticmethod
    def test_analytic() -> None:
        r"""The SNR of a signal proportional to the noise should match the analytic result.

        $$\rho = c \sqrt{T_{\text{obs}} (f_{\text{max}} - f_{\text{min}})}$$
        """
        f = np.linspace(1e-4, 1e-2, 101)
        noise = omega_noise_h2(f)
        obs_time = 1e7
        for c in (0.5, 1., 3.):
            snr, _, _ = signal_to_noise_ratio(f, c * noise, noise=noise, obs_time=obs_time)
            np.testing.assert_allclose(snr, c * np.sqrt(obs_time * (f[-1] - f[0])), rtol=1e-12)

    @staticmethod
    def test_reference() -> None:
        """The SNR should match the reference values computed with PTtools 024f993."""
        np.testing.assert_allclose(signal_to_noise_ratio(F, SIGNAL)[0], 547.6748792523158, rtol=1e-12)
        np.testing.assert_allclose(
            signal_to_noise_ratio(F, SIGNAL, noise_eb=False, noise_gb=False)[0], 780.203962546154, rtol=1e-12)
        np.testing.assert_allclose(
            signal_to_noise_ratio(F, SIGNAL, noise=omega_noise_h2(F_NOISE), f_noise=F_NOISE)[0],
            547.6455769943389, rtol=1e-12
        )

    @staticmethod
    def test_strided() -> None:
        """Strided (non-contiguous) arrays should give the same results as contiguous copies of them."""
        f = F[::2]
        signal = SIGNAL[::2]
        assert not f.flags.c_contiguous
        cases: tuple[tuple[np.ndarray | None, np.ndarray | None], ...] = (
            (None, None),
            (omega_noise_h2(F)[::2], None),
            (omega_noise_h2(F_NOISE)[::2], F_NOISE[::2]),
        )
        for noise, f_noise in cases:
            snr, f2, noise2 = signal_to_noise_ratio(f, signal, noise=noise, f_noise=f_noise)
            snr_ref, f2_ref, noise2_ref = signal_to_noise_ratio(
                f.copy(), signal.copy(),
                noise=None if noise is None else noise.copy(),
                f_noise=None if f_noise is None else f_noise.copy()
            )
            np.testing.assert_allclose(snr, snr_ref, rtol=1e-14)
            np.testing.assert_array_equal(f2, f2_ref)
            np.testing.assert_array_equal(noise2, noise2_ref)
