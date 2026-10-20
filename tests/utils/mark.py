"""Mark tests for e.g. skipping or expected failure."""

from __future__ import annotations

import unittest

import pytest

from pttools.speedup.options import NUMBA_DISABLE_JIT, USE_THREADS
import pttools.type_hints as th
from pttools.utils.system import IS_GITHUB_ACTIONS, IS_OSX, IS_WINDOWS

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.utils.decorators import TypePreservingDecorator

#: Mark tests that will fail with NUMBA_DISABLE_JIT when threads are not used.
#: The models then have to be pickled to be passed to the worker processes or interpreters.
mark_xfail_pickle_without_jit: pytest.MarkDecorator = pytest.mark.xfail(
    NUMBA_DISABLE_JIT and not USE_THREADS,
    reason="Non-bag models cannot be pickled without Numba."
)
#: Mark slow tests to be skipped on GitHub Actions Windows and macOS runners, and when Numba JIT is disabled,
#: as the tests would then be much slower
skip_slow: TypePreservingDecorator = unittest.skipIf(
    (IS_GITHUB_ACTIONS and (IS_OSX or IS_WINDOWS)) or NUMBA_DISABLE_JIT,
    reason="This test is slow and would consume a lot of runner time, especially when Numba JIT is disabled."
)
#: Mark tests that use parallel executors (threads, processes or interpreters),
#: so that they will be launched from the same runner.
uses_parallel: pytest.MarkDecorator = pytest.mark.xdist_group(name="parallel")
