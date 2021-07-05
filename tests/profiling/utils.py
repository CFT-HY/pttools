import abc
import os


PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test-results",
    "profiles"
)
if not os.path.isdir(PROFILE_DIR):
    os.mkdir(PROFILE_DIR)


class Profiler(abc.ABC):
    def __init__(self, name: str):
        self.name = name
