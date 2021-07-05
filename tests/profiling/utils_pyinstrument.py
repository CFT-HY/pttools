import os

import pyinstrument

from . import utils

PROFILE_DIR = os.path.join(utils.PROFILE_DIR, "pyinstrument")
if not os.path.isdir(PROFILE_DIR):
    os.mkdir(PROFILE_DIR)


class PyInstrumentProfiler(utils.Profiler):
    def __init__(self, name: str):
        super().__init__(name)
        self.profiler = pyinstrument.Profiler()

    def __enter__(self):
        self.profiler.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop()
        process(self.profiler, self.name)


def process(profiler: pyinstrument.Profiler, name: str):
    path = os.path.join(PROFILE_DIR, f"{name}")
    print(profiler.output_text(unicode=True, color=True))

    with open(f"{path}.txt", "w") as file:
        file.write(profiler.output_text(unicode=True, color=False))

    with open(f"{path}.html", "w") as file:
        file.write(profiler.output_html())
