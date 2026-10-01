"""Unit tests for the speedup module."""

import contextlib
import importlib
import importlib.metadata
import io
import os
import subprocess
import sys
import typing as tp
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from pttools import speedup
from pttools.analysis import save_fig
from pttools.speedup import njit, spline, tbb
from pttools.speedup.parallel import parallel_debug_message, run_parallel
import pttools.type_hints as th
from pttools.utils import assert_allclose
from tests.utils import TEST_FIGURE_PATH

TBB_INSTALLED: bool
try:
    importlib.metadata.distribution("tbb")
    TBB_INSTALLED = True
except importlib.metadata.PackageNotFoundError:
    TBB_INSTALLED = False


@njit
def jitted_spline(
        x: th.FloatArr1D,
        tck: tuple[th.FloatArr1D, th.FloatArr1D, int],
        der: int = 0,
        ext: tp.Literal[0, 1, 2, 3] = 0) -> th.FloatArr1D:
    """JIT-compiled version of splev, which uses the Numba overload defined in the speedup module."""
    return scipy.interpolate.splev(x, tck, der, ext)


class TestSpeedup(unittest.TestCase):
    """Test the functions in the speedup module."""

    @staticmethod
    def test_gradient() -> None:
        arr = np.logspace(1, 5, 10)
        assert_allclose(speedup.gradient(arr), np.gradient(arr))

    @staticmethod
    def test_logspace() -> None:
        assert_allclose(speedup.logspace(1, 5, 10), np.logspace(1, 5, 10))

    @staticmethod
    def test_parallel_debug() -> None:
        parallel_debug_message("test")

    @staticmethod
    def test_run_parallel_single_output_dtype() -> None:
        params = np.array([1., 2., 3.])
        res = run_parallel(np.square, params, output_dtypes=(np.float64,), single_thread=True)
        assert isinstance(res, np.ndarray)
        assert res.dtype == np.float64
        assert_allclose(res, params**2)

    @staticmethod
    @unittest.expectedFailure
    def test_spline() -> None:
        x = np.linspace(0, 2*np.pi, 20)
        x2 = np.linspace(0, 2*np.pi, 40)
        y = np.sin(x)
        spl = scipy.interpolate.splrep(x, y, s=0)
        ref = scipy.interpolate.splev(x2, spl)
        data = spline.splev(x2, spl)

        fig: plt.Figure = plt.Figure()
        ax: plt.Axes = fig.add_subplot()
        ax.plot(x2, data, label="data")
        ax.plot(x2, ref, label="ref", ls=":")
        ax.legend()
        save_fig(fig, TEST_FIGURE_PATH / "spline_fitpack")
        plt.close(fig)

        assert_allclose(data, ref)

    @staticmethod
    def test_spline_linear() -> None:
        """Test the Numba JIT-compiled version of splev."""
        x = np.linspace(0, 2*np.pi, 10)
        x2 = np.linspace(0, 2*np.pi, 20)
        y = np.cos(x)
        spl = scipy.interpolate.splrep(x, y, k=1, s=0)
        ref = scipy.interpolate.splev(x2, spl)
        data = jitted_spline(x2, spl)

        fig: plt.Figure = plt.Figure()
        ax: plt.Axes = fig.add_subplot()
        ax.plot(x2, data, label="data")
        ax.plot(x2, ref, label="ref", ls=":")
        ax.legend()
        save_fig(fig, TEST_FIGURE_PATH / "spline_linear")
        plt.close(fig)

        try:
            assert_allclose(data, ref)
        except AssertionError as e:
            t, c, k = spl
            with np.printoptions(
                    edgeitems=30, linewidth=200,
                    formatter={"float": lambda f: f"{f:.4e}" if f < 0 else f" {f:.4e}"}):
                print("x:", x)
                print("y:", y)
                print("t:", t)
                print("c:", c)
                print("k:", k)
            raise e


@unittest.skipUnless(TBB_INSTALLED, "The tbb package is not installed.")
class TestTBB(unittest.TestCase):
    """Test that the TBB library of the tbb package is found for Numba."""

    def test_load_tbb(self) -> None:
        version = tbb.load_tbb()
        self.assertIsNotNone(version)
        self.assertGreaterEqual(version, tbb.TBB_MIN_VERSION)

    @staticmethod
    def test_numba_tbb_layer() -> None:
        # Importing the TBB extension of Numba fails if the loader cannot find the TBB library.
        importlib.import_module("numba.np.ufunc.tbbpool")
        # This is the check that Numba runs before using the TBB threading layer.
        importlib.import_module("numba.np.ufunc.parallel")._check_tbb_version_compatible()  # noqa: SLF001

    def test_tbb_version(self) -> None:
        """The TBB library is loaded on import."""
        self.assertIsNotNone(tbb.TBB_VERSION)
        self.assertGreaterEqual(tbb.TBB_VERSION, tbb.TBB_MIN_VERSION)

    def test_main_module(self) -> None:
        """The TBB check can be run with "python -m pttools.speedup.tbb" without warnings."""
        # PTtools initialises colorama on import, which writes a reset escape code to stdout at exit
        # if stdout is a terminal. Colorama treats stdout as a terminal whenever PYCHARM_HOSTED is set,
        # even if it's a pipe, so the variable is removed to get the same output in PyCharm as elsewhere.
        env = {key: value for key, value in os.environ.items() if key != "PYCHARM_HOSTED"}
        result = subprocess.run(
            [sys.executable, "-m", "pttools.speedup.tbb"], check=False, capture_output=True, text=True, env=env
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"^TBB version: \d+$")
        self.assertNotIn("Warning", result.stderr)


class TestTBBMain(unittest.TestCase):
    """Test the exit code of the TBB check, which is run with "python -m pttools.speedup.tbb"."""

    NOTE: tp.ClassVar[str] = "TBB is not available for the current CPU architecture"

    @staticmethod
    def run_main(version: int | None, is_x86_64: bool = True) -> tuple[int, str]:
        tbb_main = importlib.import_module("pttools.speedup.tbb.__main__")
        stdout = io.StringIO()
        with mock.patch.object(tbb_main, "TBB_VERSION", version), \
                mock.patch.object(tbb_main, "IS_X86_64", is_x86_64), \
                contextlib.redirect_stdout(stdout):
            returncode = tbb_main.main()
        return returncode, stdout.getvalue()

    def test_main_compatible(self) -> None:
        self.assertEqual(self.run_main(tbb.TBB_MIN_VERSION), (0, f"TBB version: {tbb.TBB_MIN_VERSION}\n"))

    def test_main_too_old(self) -> None:
        self.assertEqual(self.run_main(tbb.TBB_MIN_VERSION - 1), (1, f"TBB version: {tbb.TBB_MIN_VERSION - 1}\n"))

    def test_main_not_found(self) -> None:
        self.assertEqual(self.run_main(None), (1, "TBB version: None\n"))

    def test_main_not_found_other_architecture(self) -> None:
        """TBB is not required on CPU architectures for which the tbb package is not available."""
        returncode, output = self.run_main(None, is_x86_64=False)
        self.assertEqual(returncode, 0)
        self.assertTrue(output.startswith("TBB version: None\n"), output)
        self.assertIn(self.NOTE, output)

    def test_main_too_old_other_architecture(self) -> None:
        returncode, output = self.run_main(tbb.TBB_MIN_VERSION - 1, is_x86_64=False)
        self.assertEqual(returncode, 0)
        self.assertIn(self.NOTE, output)

    def test_main_compatible_other_architecture(self) -> None:
        """If a compatible TBB library is found, e.g. from the operating system, the note is not printed."""
        self.assertEqual(
            self.run_main(tbb.TBB_MIN_VERSION, is_x86_64=False), (0, f"TBB version: {tbb.TBB_MIN_VERSION}\n")
        )
