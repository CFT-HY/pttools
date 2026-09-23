"""Utilities for profiling."""

import abc
import os
import types

from tests.utils import TEST_RESULT_PATH

PROFILE_DIR: str = os.path.join(TEST_RESULT_PATH, "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)


class Profiler(abc.ABC):
    """Base class for profilers."""

    def __init__(self, name: str, print_to_console: bool = False) -> None:
        self.name: str = name
        self.print_to_console: bool = print_to_console

    @abc.abstractmethod
    def __enter__(self) -> None:
        """Start the profiler."""

    @abc.abstractmethod
    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None) -> None:
        """Stop the profiler and process its results."""
