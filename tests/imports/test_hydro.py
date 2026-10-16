r"""Tests for importing the data of hydrodynamic lattice simulations."""

from pathlib import Path
import tempfile
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.imports.hydro import (
    CORREIA_2025_RUNS,
    HydroRun,
    Quantity,
    data_dir,
    integral_scale,
    load_correia_2025_run,
    mean_square,
    parse_input_file,
    read_globals,
)
from pttools.omgw0 import const, freq
from pttools.omgw0.base import Omgw0Base

#: Columns of the ``.out`` file that are used by the importer
OUT_COLUMNS: tuple[str, ...] = (
    "step", "time", "rest E sym", "rest E broken", "kin fluid sym", "kin fluid broken",
    "pressure sym", "pressure broken", "N broken", "N links",
)


class SyntheticRunTest(unittest.TestCase):
    """Tests with a small synthetic run directory with known values."""

    L: tp.ClassVar[int] = 64
    n_bubbles: tp.ClassVar[int] = 8
    dt: tp.ClassVar[float] = 0.2
    epsilon_0: tp.ClassVar[float] = 0.5
    enthalpy: tp.ClassVar[float] = 0.6
    tmpdir: tp.ClassVar[tempfile.TemporaryDirectory]
    hydro_run: tp.ClassVar[HydroRun]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        path = Path(cls.tmpdir.name)
        (path / "input-test-largerun").write_text(
            f"# Comment\nLx {cls.L}\nLy {cls.L}\nLz {cls.L}\ndx 1.0\ndt {cls.dt} # time step\n"
            f"bubbles {cls.n_bubbles}\nTconst 0.25\n\n",
            encoding="utf-8")
        volume = cls.L**3
        steps = np.arange(0, 1001, 50)
        rows = []
        for step in steps:
            # rest E + pressure = enthalpy * volume, rest E = epsilon_0 * volume at step 0
            rest_e = cls.epsilon_0 * volume
            pressure = (cls.enthalpy - cls.epsilon_0) * volume
            rows.append([step, step * cls.dt, rest_e / 2, rest_e / 2, 0.01 * volume, 0.01 * volume,
                         pressure / 2, pressure / 2, step, 100])
        lines = [",".join(OUT_COLUMNS)] + [",".join(f"{value:f}" for value in row) for row in rows]
        (path / "run_1.out").write_text("\n".join(lines) + "\n", encoding="utf-8")

        k = (np.arange(cls.L) + 0.5) * 2 * np.pi / cls.L
        counts = np.ones_like(k)
        counts[-5:] = 0
        for step in (0, 500, 1000):
            for quantity in Quantity:
                amplitude = step / 1000 if quantity == Quantity.GW_TT else 1.
                p = amplitude * k**3 * np.exp(-k)
                np.savetxt(path / f"{quantity.file_prefix}{step}.txt", np.column_stack((k, p, counts)))
        cls.hydro_run = HydroRun(path, v_wall=0.5, alpha_n=0.1)

    @classmethod
    @tp.override
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_params(self) -> None:
        """The parameters are read from the input file."""
        params = parse_input_file(self.hydro_run.path / "input-test-largerun")
        assert params["dt"] == "0.2"
        assert self.hydro_run.L == self.L
        assert self.hydro_run.n_bubbles == self.n_bubbles
        assert self.hydro_run.T_n == 0.25
        assert self.hydro_run.R_star == pytest.approx(self.L / 2)

    def test_globals(self) -> None:
        r"""$\bar{\epsilon}_0$, $H_n$, $\bar{w}$ and $\Gamma$ are computed from the ``.out`` file."""
        assert self.hydro_run.epsilon_0 == pytest.approx(self.epsilon_0)
        assert self.hydro_run.H_star == pytest.approx(np.sqrt(8 * np.pi * self.epsilon_0 / 3))
        assert self.hydro_run.r_star == pytest.approx(self.hydro_run.H_star * self.hydro_run.R_star)
        assert self.hydro_run.enthalpy(500) == pytest.approx(self.enthalpy)
        assert self.hydro_run.mean_adiabatic_index(500) == pytest.approx(self.enthalpy / self.epsilon_0)
        assert self.hydro_run.time(500) == pytest.approx(100)
        # Linear extrapolation beyond the last row
        assert self.hydro_run.time(1100) == pytest.approx(220)
        np.testing.assert_allclose(self.hydro_run.ubarf_global, np.sqrt(0.02 / self.enthalpy))

    def test_steps(self) -> None:
        """The available time steps are found from the file names."""
        assert self.hydro_run.steps(Quantity.X_DIV).tolist() == [0, 500, 1000]

    def test_empty_bins_removed(self) -> None:
        """The bins without modes are removed."""
        k, _, counts = self.hydro_run.power_spectrum(Quantity.X_DIV, 500)
        assert k.size == self.L - 5
        assert np.all(counts > 0)

    def test_rms(self) -> None:
        r"""$\bar{U}$ is computed from $X$ normalised by $\bar{w}$."""
        k, p, _ = self.hydro_run.power_spectrum(Quantity.X_DIV, 500)
        assert self.hydro_run.rms(Quantity.X_DIV, 500) == pytest.approx(np.sqrt(mean_square(k, p) / self.enthalpy))
        assert self.hydro_run.rms(Quantity.VEL_DIV, 500) == pytest.approx(np.sqrt(mean_square(k, p)))
        spectrum = self.hydro_run.spectrum(500)
        assert spectrum.ubar_par == pytest.approx(self.hydro_run.rms(Quantity.X_DIV, 500), rel=1e-3)

    def test_mean_square_analytic(self) -> None:
        r"""For $\mathcal{P} = k^3 e^{-k}$, $\int \frac{dk}{k} \mathcal{P} = 2$ and $\xi = \Gamma(2) / 2 = 1/2$."""
        k = np.linspace(1e-4, 60, 100000)
        p = k**3 * np.exp(-k)
        assert mean_square(k, p) == pytest.approx(2, rel=1e-6)
        assert integral_scale(k, p) == pytest.approx(0.5, rel=1e-6)

    def test_gw_rate(self) -> None:
        r"""The GW power growth rate is $\Delta \mathcal{P}_\text{gw} / (\bar{\epsilon}_0 H_n \Delta t H_n R_*)$."""
        spectrum = self.hydro_run.spectrum(1000, 500)
        k, p, _ = self.hydro_run.power_spectrum(Quantity.GW_TT, 1000)
        delta_t = 500 * self.dt
        run = self.hydro_run
        expected = 0.5 * p / self.epsilon_0 / (run.H_star * delta_t) / (run.H_star * run.R_star)
        np.testing.assert_allclose(spectrum.pow_gw_rate, expected)
        np.testing.assert_allclose(spectrum.y, k * self.hydro_run.R_star)
        assert self.hydro_run.omega_gw(1000) == pytest.approx(mean_square(k, p) / self.epsilon_0)

    def test_spectrum_omgw0_base(self) -> None:
        """By default, the Hubble rate and the bubble spacing of the spectrum should be those of the run."""
        spectrum = self.hydro_run.spectrum(500)
        assert isinstance(spectrum, Omgw0Base)
        assert spectrum.r_star == self.hydro_run.r_star
        assert spectrum.R_star == self.hydro_run.R_star
        assert spectrum.H_star == pytest.approx(self.hydro_run.H_star, rel=1e-15)
        np.testing.assert_array_equal(spectrum.pow_gw, self.hydro_run.pow_normalised(Quantity.GW_TT, 500)[1])

    def test_spectrum_r_star(self) -> None:
        r"""The GW power spectrum should be scaled with $r_*^2$, and the other spectra should not change."""
        r_star = 0.1
        ref = self.hydro_run.spectrum(1000, 500)
        spectrum = self.hydro_run.spectrum(1000, 500, r_star=r_star)
        assert spectrum.r_star == r_star
        assert spectrum.R_star == ref.R_star
        assert spectrum.H_star * spectrum.R_star == pytest.approx(r_star, rel=1e-15)
        np.testing.assert_allclose(spectrum.pow_gw, (r_star / self.hydro_run.r_star)**2 * ref.pow_gw, rtol=1e-15)
        np.testing.assert_array_equal(spectrum.y, ref.y)
        np.testing.assert_array_equal(spectrum.pow_U_par, ref.pow_U_par)
        np.testing.assert_array_equal(spectrum.pow_gw_rate, ref.pow_gw_rate)
        with pytest.raises(ValueError, match="r_star must be positive"):
            self.hydro_run.spectrum(500, r_star=-1.)

    def test_spectrum_omgw0(self) -> None:
        r"""The conversion to $\Omega_{\text{gw},0}$ should use the given $r_*$, $T_*$, $g_*$ and $g_{s,*}$."""
        spectrum = self.hydro_run.spectrum(1000, r_star=0.1, T_star=200., g_star=90., gs_star=100.)
        assert (spectrum.T_star, spectrum.g_star, spectrum.gs_star) == (200., 90., 100.)
        assert spectrum.f_star0 == freq.f_star0(T_star=200., ge_star=spectrum.ge_star, gs_star=100.)
        np.testing.assert_allclose(spectrum.f(), freq.f(z=spectrum.y, r_star=0.1, f_star0=spectrum.f_star0))
        np.testing.assert_allclose(spectrum.omgw0_h2(), spectrum.F_gw0_h2() * spectrum.pow_gw)
        assert spectrum.snr()[0] > 0
        # Defaults
        spectrum = self.hydro_run.spectrum(1000, g_star=80.)
        assert (spectrum.T_star, spectrum.g_star, spectrum.gs_star) == (const.DEFAULT_T_STAR, 80., 80.)

    def test_read_globals_incomplete_last_row(self) -> None:
        """The incomplete last row of an interrupted run should be ignored."""
        path = self.hydro_run.path / "run_1.out"
        lines = path.read_text(encoding="utf-8").splitlines()
        truncated = Path(self.tmpdir.name) / "truncated.out"
        truncated.write_text("\n".join([*lines, lines[-1].rsplit(",", 3)[0] + ","]) + "\n", encoding="utf-8")
        ref = read_globals(path)
        data = read_globals(truncated)
        assert data.keys() == ref.keys()
        for key, values in ref.items():
            np.testing.assert_array_equal(data[key], values)

    def test_gw_rate_requires_step_prev(self) -> None:
        """The growth rate cannot be computed without the earlier time step."""
        with pytest.raises(ValueError, match="step_prev"):
            _ = self.hydro_run.spectrum(1000).pow_gw_rate


@pytest.mark.skipif(data_dir() is None, reason="The data of Correia et al. is not available.")
class CorreiaDataTest(unittest.TestCase):
    r"""Tests with the data of :correia_2025:`\ `, which are skipped if the data is not available."""

    def test_table_iv(self) -> None:
        r"""The maximum of $\bar{U}_\parallel$ and the shock time agree with :correia_2025:`\ ` table IV."""
        for name, ubar_max, t_sh in (("detonation", 0.35, 2920), ("deflagration", 0.21, 4958)):
            with self.subTest(name=name):
                run = load_correia_2025_run(name)
                assert run.R_star == pytest.approx(1024)
                assert run.ubar_par_max[1] == pytest.approx(ubar_max, abs=0.005)
                assert run.t_sh == pytest.approx(t_sh, rel=0.02)

    def test_artifact(self) -> None:
        r"""The GW power growth rate of fig. 5 has its peak at $kR_* \approx 4$-$7$ and a separate high-$k$ peak."""
        from pttools.ssm.suppression import lattice_artifact_start  # noqa: PLC0415
        for name, meta in CORREIA_2025_RUNS.items():
            with self.subTest(name=name):
                spectrum = load_correia_2025_run(name).spectrum(meta.step, meta.step_prev)
                rate = spectrum.pow_gw_rate
                assert 3 < spectrum.y[np.argmax(rate)] < 8
                i_artifact = lattice_artifact_start(spectrum.y, rate)
                assert i_artifact is not None
                assert 300 < spectrum.y[i_artifact] < 3000


if __name__ == "__main__":
    unittest.main()
