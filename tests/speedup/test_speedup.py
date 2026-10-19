"""Unit tests for the speedup module."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import typing as tp
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.interpolate

from pttools import speedup
from pttools.math import assert_allclose
from pttools.plot import save_fig
from pttools.speedup import njit, parallel, spline
from pttools.speedup.parallel import FakeExecutor, get_pool, parallel_debug_message, run_parallel
import pttools.type_hints as th
from pttools.utils.system import SUPPORTS_INTERPRETER_POOL
from tests.utils import TEST_FIGURE_PATH
from tests.utils.mark import uses_parallel


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
    @uses_parallel
    def test_get_pool_priority() -> None:
        """Test the order of priority of single_thread, use_threads, use_interpreters and processes in get_pool."""
        with get_pool(single_thread=True, use_threads=True, use_interpreters=True) as pool:
            assert isinstance(pool, FakeExecutor)
        with get_pool(use_threads=True, use_interpreters=True) as pool:
            assert isinstance(pool, ThreadPoolExecutor)
        with get_pool(use_threads=False, max_workers=1) as pool:
            assert isinstance(pool, ProcessPoolExecutor)

    @staticmethod
    @unittest.skipUnless(SUPPORTS_INTERPRETER_POOL, "InterpreterPoolExecutor requires Python 3.14 or later.")
    @uses_parallel
    def test_get_pool_interpreters() -> None:
        """Test that use_interpreters gives an InterpreterPoolExecutor that runs functions."""
        from concurrent.futures import InterpreterPoolExecutor  # noqa: PLC0415
        with get_pool(use_threads=False, use_interpreters=True, max_workers=2) as pool:
            assert isinstance(pool, InterpreterPoolExecutor)
            assert pool.submit(pow, 2, 10).result() == 1024

    @staticmethod
    def test_get_pool_interpreters_unsupported() -> None:
        """Test that use_interpreters raises an error if InterpreterPoolExecutor is not available."""
        with mock.patch.object(parallel, "SUPPORTS_INTERPRETER_POOL", False), \
                pytest.raises(NotImplementedError, match=r"Python 3\.14"), \
                get_pool(use_threads=False, use_interpreters=True):
            pass

    @staticmethod
    @uses_parallel
    def test_run_parallel_threads() -> None:
        """Test that run_parallel with threads runs the function in worker threads and gives the same results."""
        params = np.arange(12, dtype=np.float64).reshape(3, 4)
        res = run_parallel(np.square, params, output_dtypes=(np.float64,), use_threads=True, max_workers=4)
        assert isinstance(res, np.ndarray)
        assert res.shape == params.shape
        assert_allclose(res, params**2)

        names = run_parallel(
            lambda _param: threading.current_thread().name, params,
            output_dtypes=(object,), use_threads=True, max_workers=4)
        assert isinstance(names, np.ndarray)
        assert all(name.startswith("ThreadPoolExecutor") for name in names.flat)

    @staticmethod
    @uses_parallel
    def test_run_parallel_threads_multiple() -> None:
        """Test run_parallel with threads for multiple parameters and multiple outputs."""
        params = np.array([[1., 2.], [3., 4.], [5., 6.]])
        sums, prods = run_parallel(
            lambda a, b: (a + b, a * b), params,
            multiple_params=True, unpack_params=True,
            output_dtypes=(np.float64, np.float64), use_threads=True)
        assert_allclose(sums, params.sum(axis=1))
        assert_allclose(prods, params.prod(axis=1))

    @staticmethod
    def test_run_parallel_threads_single_thread() -> None:
        """Test that single_thread takes precedence over use_threads."""
        params = np.array([1., 2.])
        names = run_parallel(
            lambda _param: threading.current_thread().name, params,
            output_dtypes=(object,), use_threads=True, single_thread=True)
        assert isinstance(names, np.ndarray)
        assert all(name == threading.current_thread().name for name in names.flat)

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
