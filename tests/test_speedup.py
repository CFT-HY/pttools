"""Unit tests for the speedup module."""

import importlib
import importlib.metadata
import typing as tp
import unittest

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
