import abc
import os

from tests import test_utils

PROFILE_DIR = os.path.join(test_utils.TEST_RESULT_PATH, "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)


class Profiler(abc.ABC):
    def __init__(self, name: str):
        self.name = name
