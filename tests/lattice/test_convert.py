"""Tests for the conversion of lattice simulation runs to HDF5 files and the 2D arrays of the power spectra."""

from __future__ import annotations

from pathlib import Path
import tempfile
import typing as tp
import unittest

import h5py
import numpy as np
import pytest

from pttools.lattice import (
    FORMAT_VERSION,
    HDF5_FILE_NAME,
    INPUT_FILE_NAME,
    LatticeRun,
    Quantity,
    convert_run,
    integral_scale,
    load_run,
    mean_square,
)
from pttools.ssm.nucleation import NucType
from tests.lattice.test_lattice import OUT_COLUMNS

#: Side length of the synthetic lattice
L: int = 16
#: Parameters of the test runs that are not in the input file
RUN_KWARGS: dict[str, tp.Any] = {"v_wall": 0.5, "alpha_n": 0.1, "nuc_type": NucType.SIMULTANEOUS}
#: The time steps of the synthetic run
STEPS: tuple[int, ...] = (0, 100, 200, 300)


def write_run(
        path: Path,
        steps: tp.Sequence[int] = STEPS,
        quantities: tp.Iterable[Quantity] = (Quantity.X_DIV, Quantity.VEL_DIV, Quantity.GW_TT),
        out: bool = True) -> None:
    """Write the files of a small synthetic run, whose power spectra depend on the time step."""
    (path / INPUT_FILE_NAME).write_text(
        f"Lx {L}\nLy {L}\nLz {L}\ndx 1.0\ndt 0.5\nbubbles 2\nTconst 0.25\n", encoding="utf-8")
    if out:
        rows = [[step, 0.5 * step, 1., 1., 0., 0., 0.1, 0.1, step, 10] for step in steps]
        lines = [",".join(OUT_COLUMNS)] + [",".join(f"{value:f}" for value in row) for row in rows]
        (path / "run_1.out").write_text("\n".join(lines) + "\n", encoding="utf-8")
    k = (np.arange(L) + 0.5) * 2 * np.pi / L
    counts = np.ones_like(k)
    counts[-2:] = 0
    for i_quantity, quantity in enumerate(quantities):
        for step in steps:
            p = (1 + i_quantity) * step / 100 * k**2 * np.exp(-k)
            np.savetxt(path / f"{quantity.file_prefix}{step}.txt", np.column_stack((k, p, counts)))


class ConvertTest(unittest.TestCase):
    """Tests for the conversion and the loading of the HDF5 files."""

    @tp.override
    def setUp(self) -> None:
        self.tmpdir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self.path: Path = Path(self.tmpdir.name)

    @tp.override
    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_convert(self) -> None:
        """The HDF5 file contains the wavenumbers once, and the power spectra as 2D arrays."""
        write_run(self.path)
        h5_path = convert_run(self.path)
        assert h5_path == self.path / HDF5_FILE_NAME
        data = load_run(self.path)
        k, p, counts = np.loadtxt(self.path / f"{Quantity.GW_TT.file_prefix}200.txt", unpack=True)
        np.testing.assert_array_equal(data.k, k)
        np.testing.assert_array_equal(data.counts, counts)
        assert set(data.steps) == {Quantity.X_DIV, Quantity.VEL_DIV, Quantity.GW_TT}
        np.testing.assert_array_equal(data.steps[Quantity.GW_TT], STEPS)
        power = data.power(Quantity.GW_TT)
        assert power.shape == (len(STEPS), L)
        np.testing.assert_array_equal(power[STEPS.index(200)], p)
        np.testing.assert_array_equal(data.globals["step"], STEPS)
        assert data.params["Lx"] == str(L)
        with h5py.File(h5_path, "r") as file:
            assert file.attrs["format_version"] == FORMAT_VERSION
        # No temporary files are left behind.
        assert [file.name for file in self.path.glob("*.h5")] == [HDF5_FILE_NAME]

    def test_load_without_text_files(self) -> None:
        """After the conversion, the run is loaded from the HDF5 file alone."""
        write_run(self.path)
        ref = LatticeRun(self.path, **RUN_KWARGS)
        for file in self.path.glob("*.txt"):
            file.unlink()
        (self.path / "run_1.out").unlink()
        run = LatticeRun(self.path, **RUN_KWARGS)
        np.testing.assert_array_equal(run.omega_gw(), ref.omega_gw())
        np.testing.assert_array_equal(run.rms(Quantity.X_DIV), ref.rms(Quantity.X_DIV))
        assert run.L == L
        assert run.has_globals

    def test_reconvert(self) -> None:
        """The run is converted again when requested, or when the format version differs."""
        write_run(self.path, steps=STEPS[:2])
        assert LatticeRun(self.path, **RUN_KWARGS).steps().tolist() == list(STEPS[:2])
        write_run(self.path)
        assert LatticeRun(self.path, **RUN_KWARGS).steps().tolist() == list(STEPS[:2])
        assert LatticeRun(self.path, **RUN_KWARGS, reconvert=True).steps().tolist() == list(STEPS)
        write_run(self.path, steps=STEPS[:3])
        for file in self.path.glob(f"*{STEPS[3]}.txt"):
            file.unlink()
        with h5py.File(self.path / HDF5_FILE_NAME, "r+") as file:
            file.attrs["format_version"] = FORMAT_VERSION - 1
        assert LatticeRun(self.path, **RUN_KWARGS).steps().tolist() == list(STEPS[:3])

    def test_different_wavenumbers(self) -> None:
        """The conversion fails if the wavenumbers or the numbers of modes of the files differ."""
        for column in (0, 2):
            with self.subTest(column=column):
                write_run(self.path)
                file = self.path / f"{Quantity.VEL_DIV.file_prefix}{STEPS[2]}.txt"
                data = np.loadtxt(file)
                data[3, column] *= 1.01
                np.savetxt(file, data)
                with pytest.raises(ValueError, match="differ"):
                    convert_run(self.path)
                assert not list(self.path.glob("*.h5"))

    def test_no_spectra(self) -> None:
        """A directory without power spectrum files cannot be loaded."""
        with pytest.raises(FileNotFoundError, match="No power spectrum files"):
            LatticeRun(self.path, **RUN_KWARGS, L=L, dx=1., dt=0.5, n_bubbles=2)

    def test_different_steps(self) -> None:
        """The quantities can have different time steps, and the run can be without the global quantities."""
        write_run(self.path, quantities=(Quantity.GW_TT,), out=False)
        write_run(self.path, steps=STEPS[:2], quantities=(Quantity.X_DIV,), out=False)
        run = LatticeRun(self.path, **RUN_KWARGS)
        np.testing.assert_array_equal(run.steps(Quantity.GW_TT), STEPS)
        np.testing.assert_array_equal(run.steps(Quantity.X_DIV), STEPS[:2])
        assert run.steps(Quantity.PHI).size == 0
        assert not run.has_globals
        assert run.last_step() == STEPS[-1]

    def test_last_step(self) -> None:
        """The last time step is the last one within the log of the global quantities."""
        write_run(self.path)
        # Rewrite the log of the global quantities to end before the last time step of the power spectra.
        write_run(self.path, steps=STEPS[:3], quantities=())
        run = LatticeRun(self.path, **RUN_KWARGS)
        assert run.steps().tolist() == list(STEPS)
        assert run.last_step() == STEPS[2]


class StepArrayTest(unittest.TestCase):
    """Tests for the methods that accept an array of time steps."""

    tmpdir: tp.ClassVar[tempfile.TemporaryDirectory]
    lattice_run: tp.ClassVar[LatticeRun]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        write_run(Path(cls.tmpdir.name))
        cls.lattice_run = LatticeRun(cls.tmpdir.name, **RUN_KWARGS)

    @classmethod
    @tp.override
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_power_spectrum(self) -> None:
        """An array of time steps gives the power spectra of the single time steps, and the default is all of them."""
        steps = np.array([300, 100])
        for quantity in (Quantity.X_DIV, Quantity.VEL_DIV, Quantity.GW_TT):
            with self.subTest(quantity=quantity):
                k, p_all = self.lattice_run.pow_normalised(quantity)
                assert p_all.shape == (len(STEPS), k.size)
                _, p = self.lattice_run.pow_normalised(quantity, steps)
                for i, step in enumerate(steps):
                    k_step, p_step = self.lattice_run.pow_normalised(quantity, int(step))
                    np.testing.assert_array_equal(k_step, k)
                    np.testing.assert_array_equal(p[i], p_step)
                    np.testing.assert_array_equal(p_all[STEPS.index(int(step))], p_step)

    def test_scalars(self) -> None:
        """The RMS values, integral scales and GW power of an array of time steps are those of the single steps."""
        steps = np.array(STEPS[1:])
        rms = self.lattice_run.rms(Quantity.X_DIV, steps)
        xi = self.lattice_run.integral_scale(Quantity.X_DIV, steps)
        omega = self.lattice_run.omega_gw(steps)
        for i, step in enumerate(steps):
            assert rms[i] == pytest.approx(self.lattice_run.rms(Quantity.X_DIV, int(step)), rel=1e-14)
            assert xi[i] == pytest.approx(self.lattice_run.integral_scale(Quantity.X_DIV, int(step)), rel=1e-14)
            assert omega[i] == pytest.approx(self.lattice_run.omega_gw(int(step)), rel=1e-14)
        np.testing.assert_array_equal(self.lattice_run.omega_gw()[1:], omega)

    def test_zero_spectrum(self) -> None:
        """The integral scale of a power spectrum that is zero is nan, and the RMS value is zero."""
        assert np.isnan(self.lattice_run.integral_scale(Quantity.X_DIV, 0))
        assert self.lattice_run.rms(Quantity.X_DIV, 0) == 0

    def test_missing_step(self) -> None:
        """A time step without a power spectrum gives an error."""
        with pytest.raises(ValueError, match=r"not available at the time steps \[150\]"):
            self.lattice_run.pow_normalised(Quantity.GW_TT, [100, 150])

    @staticmethod
    def test_integrals_2d() -> None:
        """The integrals of 2D power spectra are those of their rows."""
        k = np.linspace(0.1, 10, 100)
        p = np.vstack([k**2 * np.exp(-k), 2 * k**3 * np.exp(-k)])
        ms = mean_square(k, p)
        xi = integral_scale(k, p)
        assert ms.shape == xi.shape == (2,)
        for i in range(2):
            # The summation order of the integrals along an axis may differ in the last digit.
            assert ms[i] == pytest.approx(mean_square(k, p[i]), rel=1e-14)
            assert xi[i] == pytest.approx(integral_scale(k, p[i]), rel=1e-14)


if __name__ == "__main__":
    unittest.main()
