import abc
import os


PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test-results",
    "profiles"
)
os.makedirs(PROFILE_DIR, exist_ok=True)


class Profiler(abc.ABC):
    def __init__(self, name: str):
        self.name = name
