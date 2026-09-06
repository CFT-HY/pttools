"""Utilities for profiling."""

import abc
import os

from tests.utils import TEST_RESULT_PATH

PROFILE_DIR = os.path.join(TEST_RESULT_PATH, "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)


class Profiler(abc.ABC):
    """Base class for profilers."""

    def __init__(self, name: str, print_to_console: bool = False):
        self.name = name
        self.print_to_console = print_to_console

    @abc.abstractmethod
    def __enter__(self):
        """Start the profiler."""

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the profiler and process its results."""
