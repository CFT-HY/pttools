"""Mark tests for e.g. skipping or expected failure"""

import unittest

import pytest

from pttools.speedup.options import NUMBA_DISABLE_JIT
from pttools.utils.system import IS_GITHUB_ACTIONS, IS_OSX, IS_WINDOWS

#: Mark tests that will fail with NUMBA_DISABLE_JIT
mark_xfail_multiprocessing_jit = pytest.mark.xfail(
    NUMBA_DISABLE_JIT,
    reason="Multiprocessing without Numba is not supported for non-bag models."
)
#: Mark slow tests to be skipped on GitHub Actions Windows and macOS runners
skip_slow = unittest.skipIf(
    IS_GITHUB_ACTIONS and (IS_OSX or IS_WINDOWS),
    reason="This test is slow and would consume a lot of runner time."
)
#: Mark tests that use multiprocessing, so that they will be launched from the same runner.
uses_multiprocessing = pytest.mark.xdist_group(name="multiprocessing")
