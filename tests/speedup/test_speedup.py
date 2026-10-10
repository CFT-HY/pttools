"""Unit tests for the speedup module."""

import typing as tp
import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from pttools import speedup
from pttools.analysis import save_fig
from pttools.speedup import njit, spline
from pttools.speedup.parallel import parallel_debug_message, run_parallel
import pttools.type_hints as th
from pttools.utils import assert_allclose
from tests.utils import TEST_FIGURE_PATH


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
        """Test that the gradient gives the same results as np.gradient."""
        arr = np.logspace(1, 5, 10)
        assert_allclose(speedup.gradient(arr), np.gradient(arr))

    @staticmethod
    def test_logspace() -> None:
        """Test that the logspace gives the same results as np.logspace."""
        assert_allclose(speedup.logspace(1, 5, 10), np.logspace(1, 5, 10))

    @staticmethod
    def test_parallel_debug() -> None:
        """Test that the parallel debug message can be printed."""
        parallel_debug_message("test")

    @staticmethod
    def test_run_parallel_single_output_dtype() -> None:
        """Test that run_parallel with a single output dtype returns a single array of that dtype."""
        params = np.array([1., 2., 3.])
        res = run_parallel(np.square, params, output_dtypes=(np.float64,), single_thread=True)
        assert isinstance(res, np.ndarray)
        assert res.dtype == np.float64
        assert_allclose(res, params**2)

    @staticmethod
    @unittest.expectedFailure
    def test_spline() -> None:
        """Test that the spline evaluation of the speedup module gives the same results as SciPy splev."""
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
