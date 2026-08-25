"""Performance tests"""

import logging
import os
import unittest
import textwrap

from pttools.analysis import time_and_plot_threads
from pttools.speedup import DEFAULT_VARYING_NUMBA_THREADS, NUMBA_DISABLE_JIT
import pttools.type_hints as th
from pttools.utils.system import IS_GITHUB_ACTIONS, IS_WINDOWS
from tests.utils.const import TEST_RESULT_PATH
from tests.utils.mark import skip_slow

logger = logging.getLogger(__name__)

PERFORMANCE_DIR = os.path.join(TEST_RESULT_PATH, "performance")
os.makedirs(PERFORMANCE_DIR, exist_ok=True)

if NUMBA_DISABLE_JIT:
    __TEXT = "Numba JIT is disabled. Performance tests will be single-threaded."
    print(__TEXT)
    logger.warning(__TEXT)


class TestPerformance(unittest.TestCase):
    #: These tests measure the scaling of the performance with the number of threads,
    #: and should therefore not be run concurrently with each other.
    #: See the "pytest_collection_modifyitems" hook in conftest.py for details.
    RUN_ON_SAME_WORKER: bool = True

    @staticmethod
    def time_and_plot(
            name: str, filename: str, stmt: str, setup: str,
            n_iterations: int, n_threads: th.IntArr1D = DEFAULT_VARYING_NUMBA_THREADS):
        return time_and_plot_threads(
            name=name, filename=filename, path=PERFORMANCE_DIR,
            stmt=stmt, setup=setup,
            n_iterations=n_iterations, n_threads=n_threads
        )

    @classmethod
    @skip_slow
    def test_performance_bubble(cls):
        setup = textwrap.dedent("""
        from pttools.bubble import Bubble
        from pttools.models import BagModel

        model = BagModel()
        Bubble(model, v_wall=0.3, alpha_n=0.1)
        """)
        command = "Bubble(model, v_wall=0.3, alpha_n=0.1)"
        cls.time_and_plot("Bubble", "bubble", command, setup, 10)

    @classmethod
    @skip_slow
    def test_performance_bubble_and_spectrum(cls):
        setup = textwrap.dedent("""
        from pttools.bubble import Bubble
        from pttools.models import BagModel
        from pttools.omgw0 import Spectrum

        model = BagModel()
        bubble = Bubble(model, v_wall=0.3, alpha_n=0.1)
        Spectrum(bubble, r_star=0.1)
        """)
        command = textwrap.dedent("""
        bubble = Bubble(model, v_wall=0.3, alpha_n=0.1)
        Spectrum(bubble, r_star=0.1)
        """)
        cls.time_and_plot("Bubble and Spectrum", "bubble_and_spectrum", command, setup, 10)

    @classmethod
    @skip_slow
    def test_performance_gw(cls):
        setup = textwrap.dedent("""
        import numpy as np
        from pttools import ssm

        z = np.logspace(0, 2, 100)
        ssm.power_gw_bag(z, (0.1, 0.1))
        """)
        command = "ssm.power_gw_bag(z, (0.1, 0.1))"
        cls.time_and_plot("power_gw_bag", "power_gw_bag", command, setup, 10)

    @classmethod
    @skip_slow
    def test_performance_sin_transform(cls):
        setup = textwrap.dedent("""
        import numpy as np
        from pttools.ssm.sin_transform import sin_transform

        z = np.logspace(0, 2, 10000)
        xi = np.linspace(0, 1, 10000)
        # This is an arbitrary function
        f = np.amax([np.zeros_like(xi), np.cos(xi)], axis=0)
        sin_transform(z, xi, f)
        """)
        command = "sin_transform(z, xi, f)"
        cls.time_and_plot("sin_transform", "sin_transform", command, setup, 10)

    @classmethod
    @skip_slow
    def test_performance_spectrum(cls):
        setup = textwrap.dedent("""
        from pttools.bubble import Bubble
        from pttools.models import BagModel
        from pttools.omgw0 import Spectrum

        model = BagModel()
        bubble = Bubble(model, v_wall=0.3, alpha_n=0.1)
        Spectrum(bubble, r_star=0.1)
        """)
        command = "Spectrum(bubble, r_star=0.1)"
        cls.time_and_plot("Spectrum", "spectrum", command, setup, 10)


if __name__ == "__main__":
    unittest.main()
