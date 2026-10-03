"""Performance profiling script.

When implementing new tests, the functions should be called at least once to JIT-compile them before profiling
"""

import abc
import unittest

from pttools import speedup


class TestProfile(abc.ABC, unittest.TestCase):
    """Base class for performance profiling tests."""

    NAME: str

    @classmethod
    def setUpClass(cls) -> None:
        """JIT-compile the profiled code with :meth:`setup_numba`, unless Numba JIT is disabled."""
        if not speedup.NUMBA_DISABLE_JIT:
            cls.setup_numba()

    @classmethod
    @abc.abstractmethod
    def setup_numba(cls) -> None:
        """Run the command to be profiled before profiling.

        This ensures that it's already fully Numba-jitted when profiled.
        """
