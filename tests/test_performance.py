"""Performance tests"""

import logging
import os
import timeit
import unittest
import textwrap
import typing as tp

import numba
from pttools import speedup
from tests.utils.const import TEST_RESULT_PATH

logger = logging.getLogger(__name__)

#: Whether the installed Numba has support for setting the number of threads
NUMBA_HAS_GET_NUM_THREADS: bool = hasattr(numba, "get_num_threads")
PERFORMANCE_DIR = os.path.join(TEST_RESULT_PATH, "performance")
os.makedirs(PERFORMANCE_DIR, exist_ok=True)

if speedup.NUMBA_DISABLE_JIT:
    __TEXT = "Numba JIT is disabled. Performance tests will be single-threaded."
    print(__TEXT)
    logger.warning(__TEXT)


class TestPerformance(unittest.TestCase):
    @staticmethod
    def run_and_log(name: str, setup: str, command: str, number: int, num_threads: int, file: tp.TextIO = None):
        result = timeit.timeit(command, setup=setup, number=number)
        text = f"{name} performance with {num_threads} threads and {number} iterations: "\
               f"{result:.2f} s, {result/number:.3f} s/iteration"
        # Ensure output to stdout and therefore testing pipeline logs
        print(text)
        logger.info(text)
        if file is not None:
            file.write(f"{text}\n")

    @classmethod
    def run_with_different_threads(cls, name: str, setup: str, command: str, number: int):
        with open(os.path.join(PERFORMANCE_DIR, f"{name}.txt"), "w") as file:
            if speedup.NUMBA_DISABLE_JIT:
                cls.run_and_log(name, setup, command, number, 1, file)
            else:
                default_threads = numba.get_num_threads()
                numba.set_num_threads(1)
                cls.run_and_log(name, setup, command, number, 1, file)
                numba.set_num_threads(2)
                cls.run_and_log(name, setup, command, number, 2, file)
                if default_threads > 4:
                    numba.set_num_threads(4)
                    cls.run_and_log(name, setup, command, number, 4, file)
                if default_threads > 8:
                    numba.set_num_threads(8)
                    cls.run_and_log(name, setup, command, number, 8, file)
                numba.set_num_threads(default_threads)
                if default_threads > 2:
                    cls.run_and_log(name, setup, command, number, default_threads, file)
                logger.info(f"Numba threading layer used: {numba.threading_layer()}")

    @classmethod
    @unittest.skipIf(speedup.GITHUB_ACTIONS and speedup.IS_WINDOWS, reason="GitHub Actions Windows runners are slow")
    def test_performance_gw(cls):
        setup = textwrap.dedent("""
        import numpy as np
        import pttools.ssmtools as ssm

        z = np.logspace(0,2,100)
        gw = ssm.power_gw_scaled_bag(z, (0.1,0.1))
        """)
        command = "gw = ssm.power_gw_scaled_bag(z, (0.1,0.1))"
        cls.run_with_different_threads("GW", setup, command, 10)

    @classmethod
    @unittest.skipIf(speedup.GITHUB_ACTIONS and speedup.IS_WINDOWS, reason="GitHub Actions Windows runners are slow")
    def test_performance_sin_transform(cls):
        setup = textwrap.dedent("""
        import numpy as np
        import pttools.ssmtools.calculators as calc

        z = np.logspace(0, 2, 10000)
        xi = np.linspace(0, 1, 10000)
        # This is an arbitrary function
        f = np.amax([np.zeros_like(xi), np.cos(xi)], axis=0)
        """)
        command = "transformed = calc.sin_transform(z, xi, f)"
        cls.run_with_different_threads("sin_transform", setup, command, 10)


if __name__ == "__main__":
    unittest.main()
