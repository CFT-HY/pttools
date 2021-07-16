import abc
import os

from tests.utils import TEST_RESULT_PATH

PROFILE_DIR = os.path.join(TEST_RESULT_PATH, "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)


class Profiler(abc.ABC):
    def __init__(self, name: str, print_to_console: bool = False):
        self.name = name
        self.print_to_console = print_to_console
