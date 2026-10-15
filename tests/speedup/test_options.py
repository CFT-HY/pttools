"""Tests for the prediction of the Numba threading layer and the related startup warnings."""

import os
import subprocess
import sys
import unittest

import numba
import pytest

from pttools.speedup import options
from pttools.speedup import threading as speedup_threading


def import_pttools(numba_env: dict[str, str], start_method: str | None = None) -> subprocess.CompletedProcess:
    """Import PTtools in a new process with the given Numba environment variables, and print the threading layer."""
    code = (
        "import logging, multiprocessing\n"
        + ("" if start_method is None else f"multiprocessing.set_start_method({start_method!r})\n")
        + "logging.basicConfig(level=logging.WARNING)\n"
        "from pttools.speedup import options\n"
        "print(options.NUMBA_THREADING_LAYER, options.NUMBA_OPENMP_VENDOR, options.CONCURRENT_PARALLELISM)\n"
    )
    # See test_tbb.py for PYCHARM_HOSTED.
    env = {key: value for key, value in os.environ.items() if key != "PYCHARM_HOSTED"}
    env.update({"NUMBA_DISABLE_JIT": "0", **numba_env})
    return subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env)


class TestUseThreads(unittest.TestCase):
    """Tests for the default of using threads instead of processes."""

    @staticmethod
    def test_use_threads_env() -> None:
        """The environment variable PTTOOLS_USE_THREADS should set the default of use_threads."""
        code = (
            "import inspect\n"
            "from pttools.analysis.parallel import create_bubbles, create_spectra\n"
            "from pttools.speedup import options\n"
            "from pttools.speedup.parallel import get_pool, run_parallel\n"
            "funcs = (create_bubbles, create_spectra, get_pool, run_parallel)\n"
            "print(options.USE_THREADS, *(inspect.signature(f).parameters['use_threads'].default for f in funcs))\n"
        )
        env = {key: value for key, value in os.environ.items() if key != "PYCHARM_HOSTED"}
        for value, expected in (("0", "False"), ("1", "True")):
            env["PTTOOLS_USE_THREADS"] = value
            result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env)
            assert result.stdout.split()[-5:] == [expected] * 5


    @staticmethod
    def test_use_interpreters_env() -> None:
        """The environment variable PTTOOLS_USE_INTERPRETERS should set the default of use_interpreters."""
        code = (
            "import inspect\n"
            "from pttools.speedup import options\n"
            "from pttools.speedup.parallel import get_pool, run_parallel\n"
            "funcs = (get_pool, run_parallel)\n"
            "defaults = [inspect.signature(f).parameters['use_interpreters'].default for f in funcs]\n"
            "print(options.USE_INTERPRETERS, *defaults)\n"
        )
        env = {key: value for key, value in os.environ.items() if key != "PYCHARM_HOSTED"}
        for value, expected in (("1", "True"), ("0", "False")):
            env["PTTOOLS_USE_INTERPRETERS"] = value
            result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env)
            assert result.stdout.split()[-3:] == [expected] * 3


class TestThreadingLayer(unittest.TestCase):
    """Tests for the prediction of the Numba threading layer."""

    @unittest.skipIf(options.NUMBA_DISABLE_JIT, "The threading layer is not used when Numba JIT is disabled.")
    def test_prediction(self) -> None:
        """The prediction should match the threading layer that Numba selects."""
        # This initialises the threading layer.
        speedup_threading.check_numba_threading_layer(concurrent=False)
        assert speedup_threading.predict_numba_threading_layer() == numba.threading_layer()
        assert numba.threading_layer() == options.NUMBA_THREADING_LAYER
        assert (numba.threading_layer() in options.CONCURRENT_THREADING_LAYERS) == options.CONCURRENT_PARALLELISM

    @unittest.skipIf(options.NUMBA_DISABLE_JIT, "The threading layer is not used when Numba JIT is disabled.")
    def test_check_threading_layer(self) -> None:
        """The check should return the threading layer, and raise an error if it's not the expected one."""
        layer = speedup_threading.check_numba_threading_layer(concurrent=False)
        assert layer == numba.threading_layer()
        assert speedup_threading.check_numba_threading_layer(expected=layer, concurrent=False) == layer
        with pytest.raises(RuntimeError, match="Expected the Numba threading layer"):
            speedup_threading.check_numba_threading_layer(expected="nonexistent", concurrent=False)

    @staticmethod
    def test_check_threading_layer_main() -> None:
        """The check of the macOS workflow should work with a forced threading layer that supports concurrency."""
        layer = "tbb" if speedup_threading.predict_numba_threading_layer() == "tbb" else "omp"
        code = f"from pttools.speedup.threading import check_numba_threading_layer as c; print(c({layer!r}))"
        env = {key: value for key, value in os.environ.items() if key != "PYCHARM_HOSTED"}
        env.update({"NUMBA_DISABLE_JIT": "0", "NUMBA_THREADING_LAYER": layer})
        result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        assert result.stdout.split()[-1] == layer
        env["NUMBA_THREADING_LAYER"] = "workqueue"
        result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, env=env)
        assert result.returncode != 0
        assert "Expected the Numba threading layer" in result.stderr

    @staticmethod
    def test_workqueue_warning() -> None:
        """The workqueue threading layer should not support concurrent parallelism, and give a warning."""
        result = import_pttools({"NUMBA_THREADING_LAYER": "workqueue"})
        assert result.stdout.split()[-3:] == ["workqueue", "None", "False"]
        assert "does not support concurrent parallelism" in result.stderr
        assert "pttools-gw[performance]" in result.stderr

    @staticmethod
    @unittest.skipIf(speedup_threading.numba_openmp_vendor() != "GNU", "GNU OpenMP is not available.")
    def test_gnu_openmp_fork_warning() -> None:
        """GNU OpenMP with the fork start method should give a warning, but other start methods should not."""
        result = import_pttools({"NUMBA_THREADING_LAYER": "omp"}, start_method="fork")
        assert result.stdout.split()[-3:] == ["omp", "GNU", "True"]
        assert "GNU OpenMP" in result.stderr
        result = import_pttools({"NUMBA_THREADING_LAYER": "omp"}, start_method="spawn")
        assert "GNU OpenMP" not in result.stderr
        assert "concurrent parallelism" not in result.stderr


if __name__ == "__main__":
    unittest.main()
