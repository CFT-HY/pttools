"""Mark tests for e.g. skipping or expected failure"""

import unittest

import pytest

from pttools.speedup.options import NUMBA_DISABLE_JIT
from pttools.utils.system import IS_GITHUB_ACTIONS, IS_OSX, IS_WINDOWS

mark_xfail_multiprocessing_jit = pytest.mark.xfail(
    NUMBA_DISABLE_JIT,
    reason="Multiprocessing without Numba is not supported for non-bag models."
)
skip_slow = unittest.skipIf(
    IS_GITHUB_ACTIONS and (IS_OSX or IS_WINDOWS),
    reason="This test is slow and would consume a lot of runner time."
)
