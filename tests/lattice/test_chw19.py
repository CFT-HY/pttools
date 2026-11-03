r"""Tests for importing the simulations of :cutting_2019:`\ ` and the older format of the simulation code."""

from __future__ import annotations

from pathlib import Path
import tempfile
import typing as tp
import unittest

import h5py
import numpy as np
import pytest

from pttools.lattice import (
    GLOBALS_TXT_COLUMNS,
    HDF5_FILE_NAME,
    LatticeRun,
    Quantity,
    chw19,
    convert_run,
    find_input_file,
    load_run,
    read_globals,
)
from pttools.ssm.nucleation import NucType

#: Name of the synthetic run directory
RUN_NAME: str = "alph-0.1-vw-0.5-Nb-2-L-16-dx-1.0-seed-2"
#: Side length of the synthetic lattice
L: int = 16
#: Rows ``step x y z`` of the bubble location file of the synthetic run
BUBBLE_LOCS: tuple[tuple[int, int, int, int], ...] = ((0, 0, 0, 0), (0, 3, 5, 7))
#: The time steps of the synthetic run
STEPS: tuple[int, ...] = (0, 100, 200, 300)
#: $\bar{\epsilon}_0$ of the synthetic run
EPSILON_0: float = 0.5
#: $\bar{w}$ of the synthetic run after the first row, which has zero pressure as in the simulations
ENTHALPY: float = 0.6


def write_globals_txt(path: Path, n_cols: int = 15, steps: tp.Sequence[int] = STEPS) -> None:
    """Write a synthetic ``globals.txt`` file with the given number of columns."""
    volume = L**3
    columns = GLOBALS_TXT_COLUMNS[n_cols]
    rows = []
    for step in steps:
        values = dict.fromkeys(columns, 0.)
        values |= {
            "step": step, "time": 0.5 * step, "total E": EPSILON_0 * volume, "kin fluid": 0.01 * volume,
            "GW Energy": 1e-6 * step, "Nb": 2, "rest E": EPSILON_0 * volume,
            "pressure": 0. if step == 0 else (ENTHALPY - EPSILON_0) * volume, "gamma_max": 1.,
        }
        rows.append("\t".join(f"{values[column]:f}" for column in columns))
    (path / "globals.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_legacy_run(path: Path, steps: tp.Sequence[int] = STEPS) -> None:
    """Write the files of a small synthetic run in the format of the older version of the simulation code."""
    path.mkdir(parents=True, exist_ok=True)
    input_text = f"Lx {L}\nLy {L}\nLz {L}\ndx 1.0\ndt 0.5\nbubbles 2\nTconst 0.25\ngstar 106.75\n"
    (path / "input-test-BAG").write_text(input_text, encoding="utf-8")
    # A backup file of a text editor, which should be ignored
    (path / "input-test-BAG~").write_text(input_text.replace("dt 0.5", "dt 0.1"), encoding="utf-8")
    write_globals_txt(path, steps=steps)
    np.savetxt(path / "bubblelocs.txt", np.array(BUBBLE_LOCS), fmt="%d")
    k = (np.arange(L) + 0.5) * 2 * np.pi / L
    counts = np.ones_like(k)
    for i_prefix, prefix in enumerate(("power-step", "tot-ps-step", "div-ps-step", "rot-ps-step")):
        for step in steps:
            p = (1 + i_prefix) * (1 + step / 100) * k**2 * np.exp(-k)
            np.savetxt(path / f"{prefix}{step}.txt", np.column_stack((k, p, counts)))
    # An unnormalised FFT file of the older version, which should not be imported
    np.savetxt(path / "fft-e-100.txt", np.column_stack((k, 1e10 * k, counts)))
    # An empty file of an interrupted run, which should be skipped
    (path / "tot-ps-step400.txt").touch()


class GlobalsTxtTest(unittest.TestCase):
    """Tests for reading the ``globals.txt`` files."""

    @tp.override
    def setUp(self) -> None:
        self.tmpdir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self.path: Path = Path(self.tmpdir.name)

    @tp.override
    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_columns(self) -> None:
        """The columns are named by their number, and the shared columns have the same names."""
        for n_cols, columns in GLOBALS_TXT_COLUMNS.items():
            with self.subTest(n_cols=n_cols):
                write_globals_txt(self.path, n_cols)
                data = read_globals(self.path / "globals.txt")
                assert tuple(data) == columns
                assert columns[:13] == GLOBALS_TXT_COLUMNS[13]
                np.testing.assert_array_equal(data["step"], STEPS)
                assert data["rest E"][0] == pytest.approx(EPSILON_0 * L**3)

    def test_incomplete_last_row(self) -> None:
        """The incomplete last row of an interrupted run is ignored."""
        write_globals_txt(self.path)
        path = self.path / "globals.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(text + "400\t200.0\t1.0\n", encoding="utf-8")
        data = read_globals(path)
        np.testing.assert_array_equal(data["step"], STEPS)

    def test_unknown_format(self) -> None:
        """An unknown number of columns raises an error."""
        (self.path / "globals.txt").write_text("0\t0.0\t1.0\n100\t20.0\t1.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown format"):
            read_globals(self.path / "globals.txt")


class LegacyRunTest(unittest.TestCase):
    """Tests with a small synthetic run in the format of the older version of the simulation code."""

    tmpdir: tp.ClassVar[tempfile.TemporaryDirectory]
    path: tp.ClassVar[Path]
    lattice_run: tp.ClassVar[LatticeRun]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmpdir.name) / RUN_NAME
        write_legacy_run(cls.path)
        cls.lattice_run = chw19.load_chw19_run(str(cls.path))

    @classmethod
    @tp.override
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_run_params(self) -> None:
        """The parameters are taken from the name of the directory."""
        params = chw19.run_params(self.path)
        assert (params.alpha_n, params.v_wall, params.n_bubbles, params.L, params.dx) == (0.1, 0.5, 2, L, 1.)
        assert params.nuc_type == NucType.SIMULTANEOUS
        assert params.label == "CHW19 seed-2"
        assert "L=16" in params.label_unicode
        # The wall speed is not in table I.
        assert params.fig_step is None
        with pytest.raises(ValueError, match="time steps of the figures"):
            _ = params.fig_steps
        with pytest.raises(ValueError, match="pattern"):
            chw19.run_params(self.path.parent / "detonation")

    def test_chw19_runs(self) -> None:
        """The runs of a data directory are found by the names of their directories."""
        runs = chw19.chw19_runs(self.path.parent)
        assert list(runs) == [RUN_NAME]
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing"
            assert not chw19.chw19_runs(missing, missing_ok=True)
            with pytest.raises(FileNotFoundError):
                chw19.chw19_runs(missing)

    def test_fig_steps(self) -> None:
        r"""The time steps of the figures end at ${t}_\text{fin}$ and start $2 R_*$ before it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alph-0.1-vw-0.92-Nb-8-L-16-dx-1.0"
            # t_fin = 2400 and 2 R_* = 16 correspond to the time steps 4800 and 4768 with dt = 0.5.
            write_legacy_run(path, steps=(0, 4700, 4760, 4770, 4800, 4900))
            params = chw19.run_params(path)
            assert params.fig_steps == (4800, 4770)

    def test_input_file(self) -> None:
        """The input file is found with the pattern, ignoring the backup files, and multiple files are an error."""
        assert find_input_file(self.path) == self.path / "input-test-BAG"
        assert self.lattice_run.dt == 0.5
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            assert find_input_file(path) is None
            (path / "input-a").touch()
            (path / "input-b").touch()
            with pytest.raises(ValueError, match="multiple input files"):
                find_input_file(path)

    def test_legacy_spectra(self) -> None:
        """The power spectrum files of the older naming convention are imported as the corresponding quantities."""
        assert set(self.lattice_run.data.steps) == {
            Quantity.GW_TT, Quantity.VEL_TOT, Quantity.VEL_DIV, Quantity.VEL_ROT}
        # The empty file is skipped.
        np.testing.assert_array_equal(self.lattice_run.steps(Quantity.VEL_TOT), STEPS)
        _, p, _ = np.loadtxt(self.path / "div-ps-step200.txt", unpack=True)
        np.testing.assert_array_equal(self.lattice_run.power_spectrum(Quantity.VEL_DIV, 200)[1], p[1:])

    def test_duplicate_spectra(self) -> None:
        """Files of both naming conventions for the same quantity and time step are an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / RUN_NAME
            write_legacy_run(path)
            (path / f"{Quantity.VEL_DIV.file_prefix}100.txt").write_text(
                (path / "div-ps-step100.txt").read_text(encoding="utf-8"), encoding="utf-8")
            with pytest.raises(ValueError, match="multiple power spectrum files"):
                convert_run(path)

    def test_bubble_locs(self) -> None:
        """The bubble locations are stored in the HDF5 file, including the first bubble at the origin."""
        np.testing.assert_array_equal(self.lattice_run.bubble_locs, BUBBLE_LOCS)
        with h5py.File(self.path / HDF5_FILE_NAME, "r") as file:
            assert file.attrs["bubble_locs_file"] == "bubblelocs.txt"
            assert file.attrs["globals_file"] == "globals.txt"
            assert file.attrs["input_file"] == "input-test-BAG"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / RUN_NAME
            write_legacy_run(path)
            (path / "bubblelocs.txt").unlink()
            assert load_run(path).bubble_locs.shape == (0, 4)

    def test_globals(self) -> None:
        r"""The global quantities are the totals of ``globals.txt``."""
        run = self.lattice_run
        assert run.epsilon_0 == pytest.approx(EPSILON_0)
        assert run.enthalpy(200) == pytest.approx(ENTHALPY)
        np.testing.assert_allclose(run.ubarf_global[1:], np.sqrt(0.01 / ENTHALPY))
        assert run.time(200) == pytest.approx(100)
        assert chw19.epsilon_r(run) == pytest.approx(3 * np.pi**2 / 90 * 106.75 * 0.25**4)

    def test_u_quantity(self) -> None:
        r"""Without the spectra of $X^i$, $U^i$ is approximated by $v^i$."""
        run = self.lattice_run
        assert not run.has_x
        assert run.u_quantity(Quantity.X_DIV) == Quantity.VEL_DIV
        assert run.u_quantity(Quantity.X_ROT) == Quantity.VEL_ROT
        assert run.u_quantity(Quantity.GW_TT) == Quantity.GW_TT
        assert run.ubar_par_max[1] == pytest.approx(np.max(run.rms(Quantity.VEL_DIV)))
        spectrum = run.spectrum(300, 200)
        np.testing.assert_array_equal(spectrum.pow_U_par, spectrum.pow_v_par)

    def test_table_i_values(self) -> None:
        r"""The GW power growth rate of table I is the mean of $\Omega_\text{gw} / (H_n t) / (H_n R_*)$."""
        run = self.lattice_run
        t_fin = 150.
        values = chw19.table_i_values(run, t_fin=t_fin)
        e_r = chw19.epsilon_r(run)
        H_n = np.sqrt(8 * np.pi / 3 * e_r)
        # 2 R_* = 16 includes only the last row before t_fin.
        expected = 1e-6 * 300 / e_r / (H_n * t_fin) / (H_n * run.R_star)
        assert values["Omega_gw/(H_n t)/(H_n R_*)"] == pytest.approx(expected)
        assert values["Ubar_f,max"] == pytest.approx(np.sqrt(0.01 / ENTHALPY))


@pytest.mark.skipif(chw19.data_dir() is None, reason="The data of Cutting et al. is not available.")
class CuttingDataTest(unittest.TestCase):
    r"""Tests with the data of :cutting_2019:`\ `, which are skipped if the data is not available."""

    def test_runs(self) -> None:
        """The runs are found, and the time steps of the figures are those of the final $2 R_*$."""
        runs = chw19.chw19_runs()
        params = runs["alph-0.05-vw-0.24-Nb-8-L-960-dx-1.0"]
        assert (params.v_wall, params.alpha_n, params.n_bubbles, params.L, params.dx) == (0.24, 0.05, 8, 960, 1.)
        assert params.fig_steps == (24000, 19200)

    def test_bubble_locs(self) -> None:
        """The bubble locations agree with those of the log of the simulation."""
        run = chw19.load_chw19_run("alph-0.005-vw-0.44-Nb-8-L-480-dx-2.0")
        assert run.bubble_locs.shape == (8, 4)
        np.testing.assert_array_equal(run.bubble_locs[:2], [[0, 0, 0, 0], [0, 327, 449, 37]])

    def test_table_i(self) -> None:
        r"""$\bar{U}_{f,\text{max}}$ and the GW power growth rate agree with :cutting_2019:`\ ` table I.

        The values of the table have two significant figures.
        """
        cases = (
            (0.24, 0.05, 0.036, 4.6e-8),
            (0.24, 0.34, 0.066, 2.3e-7),
            (0.44, 0.005, 0.0083, 1.0e-10),
            (0.44, 0.5, 0.19, 5.8e-5),
            (0.56, 0.11, 0.14, 1.2e-5),
            (0.56, 0.67, 0.30, 3.0e-4),
            (0.82, 0.16, 0.15, 1.2e-5),
            (0.92, 0.05, 0.045, 1.5e-7),
            (0.92, 0.67, 0.34, 4.5e-4),
        )
        for v_wall, alpha_n, ubarf_max, rate in cases:
            name = f"alph-{alpha_n}-vw-{v_wall}-Nb-8-L-960-dx-1.0"
            with self.subTest(name=name):
                values = chw19.table_i_values(chw19.load_chw19_run(name))
                assert values["Ubar_f,max"] == pytest.approx(ubarf_max, rel=0.05)
                assert values["Omega_gw/(H_n t)/(H_n R_*)"] == pytest.approx(rate, rel=0.05)


if __name__ == "__main__":
    unittest.main()
