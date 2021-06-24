"""Performance testing script"""

import logging
import os
import timeit
import unittest
import textwrap

import numba

logger = logging.getLogger(__name__)

NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT", default=None)
if NUMBA_DISABLE_JIT:
    __TEXT = "Numba JIT is disabled. Performance tests will be single-threaded."
    print(__TEXT)
    logger.warning(__TEXT)


class TestPerformance(unittest.TestCase):
    @staticmethod
    def run_and_log(name: str, setup: str, command: str, number: int, num_threads: int):
        result = timeit.timeit(command, setup=setup, number=number)
        text = f"{name} performance with {num_threads} threads and {number} iterations: "\
               f"{result:.2f} s, {result/number:.3f} s/iteration"
        # Ensure output to stdout and therefore testing pipeline logs
        print(text)
        logger.info(text)

    @classmethod
    def run_with_different_threads(cls, name: str, setup: str, command: str, number: int):
        if NUMBA_DISABLE_JIT:
            cls.run_and_log(name, setup, command, number, 1)
        else:
            default_threads = numba.get_num_threads()
            numba.set_num_threads(1)
            cls.run_and_log(name, setup, command, number, 1)
            numba.set_num_threads(2)
            cls.run_and_log(name, setup, command, number, 2)
            if default_threads > 4:
                numba.set_num_threads(4)
                cls.run_and_log(name, setup, command, number, 4)
            if default_threads > 8:
                numba.set_num_threads(8)
                cls.run_and_log(name, setup, command, number, 8)
            numba.set_num_threads(default_threads)
            if default_threads > 2:
                cls.run_and_log(name, setup, command, number, default_threads)
            logger.info(f"Numba threading layer used: {numba.threading_layer()}")

    @classmethod
    def test_performance_gw(cls):
        setup = textwrap.dedent("""
        import os
        import numpy as np

        import pttools.ssmtools as ssm

        z = np.logspace(0,2,100)
        gw = ssm.power_gw_scaled(z,[0.1,0.1])
        """)
        command = "gw = ssm.power_gw_scaled(z,[0.1,0.1])"
        cls.run_with_different_threads("GW", setup, command, 20)

    @classmethod
    def test_performance_sin_transform(cls):
        setup = textwrap.dedent("""
        import os
        import numpy as np

        import pttools.ssmtools.calculators as calc

        z = np.logspace(0, 2, 10000)
        xi = np.linspace(0, 1, 10000)
        # TODO: put some better function here
        f = np.amax([np.zeros_like(xi), np.sin(xi)], axis=0)
        """)
        command = "transformed = calc.sin_transform(z, xi, f)"
        cls.run_with_different_threads("sin_transform", setup, command, 10)


if __name__ == "__main__":
    unittest.main()
