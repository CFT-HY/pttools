"""Performance profiling script

When implementing new tests, the functions should be called at least once to JIT-compile them before profiling
"""

import abc
import unittest

from pttools import speedup


class TestProfile(abc.ABC, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not speedup.NUMBA_DISABLE_JIT:
            cls.setup_numba()

    @classmethod
    @abc.abstractmethod
    def setup_numba(cls):
        """Run the command to be profiled before profiling to ensure
        that it's already fully Numba-jitted when profiled."""
        pass
