r"""Tests for the extrapolation of the time development of the quantities of lattice simulations."""

from pathlib import Path
import tempfile
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.lattice import INPUT_FILE_NAME, LatticeComparison, LatticeRun, Quantity
from pttools.lattice.asymptote import (
    DEFAULT_RUNNING_MEAN_WINDOW,
    PowGwAsymptoteFit,
    fit_pow_gw_modes,
    gw_growth_index_prediction,
)
from pttools.math.asymptote import AsymptoteFit, asymptotic_growth, fit_asymptote
from pttools.math.numerics import running_mean
from pttools.math.power_law import PowerLawFit
from pttools.ssm.nucleation import NucType
from tests.lattice.test_lattice import OUT_COLUMNS
from tests.utils.const import LOW_ACCURACY_SPECTRUM_KWARGS


class AsymptoteFunctionTest(unittest.TestCase):
    """Tests for the GW-specific fitting functions with exact synthetic data."""

    @staticmethod
    def test_fit_pow_gw_modes() -> None:
        r"""The asymptotes and indices of each wavenumber are recovered, and $\Omega_\text{par}$ is integrated."""
        t = np.linspace(0, 5000, 200)
        k = np.array([0.01, 0.02, 0.04])
        value_inf = np.array([3., 2., 1.])
        index = np.array([0.5, 1., 1.5])
        # The function diverges at t = 0, which is clipped to zero.
        with np.errstate(divide="ignore"):
            pow_gw = np.column_stack([
                np.maximum(asymptotic_growth(t, value_inf=v, t_decay=400, index=d), 0)
                for v, d in zip(value_inf, index, strict=True)])
        fit = fit_pow_gw_modes(t=t, k=k, y=10 * k, pow_gw=pow_gw, start_fraction=0.8, t_decay_guess=300)
        np.testing.assert_allclose(fit.pow_gw_inf, value_inf, rtol=1e-6)
        np.testing.assert_allclose(fit.index, index, rtol=1e-6)
        np.testing.assert_allclose(fit.t_decay, 400, rtol=1e-6)
        np.testing.assert_allclose(fit.omega_gw_partial(), np.trapezoid(pow_gw, np.log(k), axis=1))
        t_late = np.array([1e12])
        assert fit.omega_gw_partial(t_late)[0] == pytest.approx(np.trapezoid(value_inf, np.log(k)), rel=1e-3)

    @staticmethod
    def test_pow_gw_inf_lower() -> None:
        r"""The lower limit of the uncertainty of $\mathcal{P}_\text{gw}^\infty$ is bounded by the fit at its end."""
        def mode_fit(value_err: float, index: float) -> AsymptoteFit:
            return AsymptoteFit(
                value=1., value_err=value_err, t_decay=1000., t_decay_err=0., index=index, index_err=0.,
                t_start=1000., t_end=2000.)
        # The fitted value at the end is 1 - 2^(-1) = 0.5.
        fits = (mode_fit(0.1, 1.), mode_fit(3., 1.), None)
        k = np.array([0.01, 0.02, 0.03])
        fit = PowGwAsymptoteFit(y=10 * k, k=k, t=np.array([2000.]), pow_gw=np.zeros((1, 3)), fits=fits)
        # A small uncertainty is used as it is, whereas a large one is limited by the fitted value at the end.
        np.testing.assert_allclose(fit.pow_gw_inf_lower, [0.9, 0.5, np.nan])

    @staticmethod
    def test_gw_growth_index_prediction() -> None:
        r"""The prediction is $2\zeta + 2\lambda - 1$, and the uncertainty the sum of those of the terms."""
        zeta = PowerLawFit(index=1.5, index_err=0.1, t_scale=1., t_0=0., factor=-0.5)
        lambda_ = PowerLawFit(index=0.2, index_err=0.05, t_scale=1., t_0=0., factor=1.)
        prediction, err = gw_growth_index_prediction(zeta, lambda_)
        assert prediction == pytest.approx(2.4)
        assert err == pytest.approx(0.3)


class SyntheticAsymptoteRunTest(unittest.TestCase):
    r"""Tests for the extrapolation methods of the runs and spectra with a synthetic run.

    The compressional velocity has its maximum at $t_\text{max} = 200$, after which it decays with $\zeta = 1.2$,
    and the GW power approaches its asymptote with $d = 0.8$.
    """

    L: tp.ClassVar[int] = 64
    dt: tp.ClassVar[float] = 0.2
    epsilon_0: tp.ClassVar[float] = 0.5
    t_max: tp.ClassVar[float] = 200.
    zeta: tp.ClassVar[float] = 1.2
    index: tp.ClassVar[float] = 0.8
    t_decay: tp.ClassVar[float] = 300.
    tmpdir: tp.ClassVar[tempfile.TemporaryDirectory]
    lattice_run: tp.ClassVar[LatticeRun]
    k: tp.ClassVar[np.ndarray]

    @classmethod
    def ubar_amplitude(cls, t: float) -> float:
        """Amplitude of the compressional velocity, which rises linearly to its maximum and then decays."""
        if t <= cls.t_max:
            return max(t / cls.t_max, 1e-3)
        return (1 + (t - cls.t_max) / 100)**(-cls.zeta / 2)

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        path = Path(cls.tmpdir.name)
        (path / INPUT_FILE_NAME).write_text(
            f"Lx {cls.L}\nLy {cls.L}\nLz {cls.L}\ndx 1.0\ndt {cls.dt}\nbubbles 8\nTconst 0.25\n", encoding="utf-8")
        volume = cls.L**3
        steps = np.arange(0, 25001, 250)
        rest_e = cls.epsilon_0 * volume
        rows = [
            [step, step * cls.dt, rest_e / 2, rest_e / 2, 0., 0., 0., 0., step, 100]
            for step in steps
        ]
        lines = [",".join(OUT_COLUMNS)] + [",".join(f"{value:f}" for value in row) for row in rows]
        (path / "run_1.out").write_text("\n".join(lines) + "\n", encoding="utf-8")

        cls.k = (np.arange(cls.L) + 0.5) * 2 * np.pi / cls.L
        counts = np.ones_like(cls.k)
        shape = cls.k**3 * np.exp(-cls.k)
        for step in steps:
            t = step * cls.dt
            # The function diverges at t = 0, which is clipped to zero.
            with np.errstate(divide="ignore"):
                gw = max(float(asymptotic_growth(t, value_inf=1., t_decay=cls.t_decay, index=cls.index)), 0.)
            for quantity in Quantity:
                if quantity == Quantity.GW_TT:
                    p = gw * shape
                elif quantity == Quantity.X_DIV:
                    p = cls.ubar_amplitude(t)**2 * shape
                else:
                    p = shape
                # The first bin is the zero mode, which should be excluded. Its value would distort the integrals.
                p = p.copy()
                p[0] = 1e3
                np.savetxt(path / f"{quantity.file_prefix}{step}.txt", np.column_stack((cls.k, p, counts)))
        cls.lattice_run = LatticeRun(path, v_wall=0.5, alpha_n=0.1, nuc_type=NucType.SIMULTANEOUS)

    @classmethod
    @tp.override
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_t_max(self) -> None:
        r"""The time of the maximum of $\bar{U}_\parallel$ is that of the synthetic data."""
        assert self.lattice_run.t_max == pytest.approx(self.t_max)

    def test_gw_asymptote_scale(self) -> None:
        r"""The scale is $(\Gamma \bar{U}_\parallel^2)_*^2 (H_n R_*) (H_n {t}_\text{sh})$."""
        run = self.lattice_run
        kinetic = run.kinetic_energy_fraction_par(run.ubar_par_max[0])
        assert run.gw_asymptote_scale == pytest.approx(
            kinetic**2 * (run.H_star * run.R_star) * (run.H_star * run.t_sh), rel=1e-14)

    def test_gw_asymptote_summary(self) -> None:
        """The summary has the values of the fits and of the spectrum at the given time step."""
        run = self.lattice_run
        summary = run.gw_asymptote_summary(10000, step_end=20000)
        omega_fit = run.omega_gw_asymptote(step_end=20000)
        assert summary["zeta"] == run.kinetic_energy_decay_index(20000).index
        assert summary["lambda_err"] == run.integral_scale_growth_index(20000).index_err
        assert summary["2 zeta + 2 lambda - 1"] == run.gw_growth_index_prediction(20000)[0]
        assert (summary["d_tot"], summary["d_tot_err"]) == (omega_fit.index, omega_fit.index_err)
        assert summary["Omega_inf (scaled)"] == pytest.approx(omega_fit.value / run.gw_asymptote_scale, rel=1e-14)
        assert summary["t [1/T_c]"] == pytest.approx(10000 * self.dt)
        assert summary["Omega_inf / Omega(t)"] == pytest.approx(
            omega_fit.value / run.omega_gw(10000), rel=1e-12)

    def test_kinetic_energy_decay_index(self) -> None:
        r"""$\zeta$ is recovered, and $\lambda = 0$, as the shape of the spectrum does not change."""
        assert self.lattice_run.kinetic_energy_decay_index().index == pytest.approx(self.zeta, rel=1e-4)
        assert self.lattice_run.integral_scale_growth_index().index == pytest.approx(0, abs=1e-3)
        prediction, _ = self.lattice_run.gw_growth_index_prediction()
        assert prediction == pytest.approx(2 * self.zeta - 1, abs=2e-3)

    def test_omega_gw_asymptote(self) -> None:
        r"""$\Omega_\text{gw}^\infty$ and $d_\text{tot}$ are recovered from $\Omega_\text{gw}(t)$."""
        valid = self.k[1:]
        omega_inf = np.trapezoid(valid**2 * np.exp(-valid), valid) / self.epsilon_0
        # Without the running mean, the parameters of the exact data are recovered.
        exact = self.lattice_run.omega_gw_asymptote(window=1)
        assert exact.t_start == pytest.approx(3 * self.t_max)
        assert exact.value == pytest.approx(omega_inf, rel=1e-6)
        assert exact.index == pytest.approx(self.index, rel=1e-6)
        # By default, the fit is done to the running mean of the time series.
        # The running mean distorts this coarse synthetic time series, and therefore only the procedure is tested.
        fit = self.lattice_run.omega_gw_asymptote()
        t, omega = self.lattice_run.omega_gw_series()
        start = int(np.argmin(np.abs(t - 3 * self.t_max)))
        omega = running_mean(omega, DEFAULT_RUNNING_MEAN_WINDOW)
        ref = fit_asymptote(t[start:], omega[start:], t_decay_guess=self.t_max)
        assert (fit.value, fit.index, fit.t_decay) == pytest.approx((ref.value, ref.index, ref.t_decay), rel=1e-12)
        assert self.lattice_run.omega_gw_asymptote() is fit

    def test_pow_gw_asymptote(self) -> None:
        r"""The asymptotes $\mathcal{P}_\text{gw}^\infty(k)$ and the indices $d(k)$ are recovered."""
        fit = self.lattice_run.pow_gw_asymptote(y_min=1, y_max=20, start_fraction=0.8)
        assert fit.y.min() >= 1
        assert fit.y.max() <= 20
        k = fit.k
        np.testing.assert_allclose(fit.pow_gw_inf, k**3 * np.exp(-k) / self.epsilon_0, rtol=1e-6)
        np.testing.assert_allclose(fit.index, self.index, rtol=1e-6)

    def test_step_end(self) -> None:
        """The fits can be limited to the time steps up to the given one."""
        fit = self.lattice_run.omega_gw_asymptote(step_end=20000)
        assert fit.t_end == pytest.approx(20000 * self.dt)
        assert self.lattice_run.pow_gw_asymptote(y_min=1, y_max=20, step_end=20000).t[-1] == pytest.approx(4000)

    def test_spectrum_ratio(self) -> None:
        r"""The ratio $\Omega_\text{gw}^\infty / \Omega_\text{gw}(t)$ of a spectrum is that of the run at its time."""
        spectrum = self.lattice_run.spectrum(10000, r_star=0.1)
        ratio = spectrum.omega_gw_asymptote_ratio()
        assert ratio == pytest.approx(
            self.lattice_run.omega_gw_asymptote().value / self.lattice_run.omega_gw(10000), rel=1e-12)
        assert ratio > 1
        assert self.lattice_run.spectrum(10000).omega_gw_asymptote_ratio() == pytest.approx(ratio, rel=1e-12)

    def test_comparison_omega(self) -> None:
        r"""$\Omega_\text{gw}$ of the comparison are the integrals of the spectra, scaled to the asymptote as given."""
        run = self.lattice_run
        ssm = run.ssm_spectrum(y=np.logspace(-1, 3, 200), r_star=0.1, **LOW_ACCURACY_SPECTRUM_KWARGS)
        comparison = LatticeComparison(run.spectrum(10000, 9750, r_star=0.1), ssm=ssm)
        omega = comparison.omega(step_end=20000)
        spectrum = comparison.spectrum
        omega_t = np.trapezoid(spectrum.pow_gw, np.log(spectrum.y))
        assert omega["Omega_gw(t)"] == pytest.approx(omega_t, rel=1e-12)
        assert omega["Omega_gw_inf"] == pytest.approx(
            spectrum.omega_gw_asymptote_ratio(step_end=20000) * omega_t, rel=1e-12)
        assert omega["Omega_gw SSM, unsuppressed"] == pytest.approx(np.trapezoid(ssm.pow_gw, np.log(ssm.y)), rel=1e-12)
        assert omega["Omega_gw SSM, suppressed"] == pytest.approx(
            comparison.suppression * omega["Omega_gw SSM, unsuppressed"], rel=1e-12)

    def test_spectrum_pow_gw_asymptote(self) -> None:
        r"""The GW power spectrum scaled to the asymptote is $\mathcal{P}_\text{gw}$ times the ratio."""
        spectrum = self.lattice_run.spectrum(10000, r_star=0.1)
        np.testing.assert_allclose(
            spectrum.pow_gw_asymptote(step_end=20000),
            spectrum.omega_gw_asymptote_ratio(step_end=20000) * spectrum.pow_gw, rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
