import os

import pyinstrument

from . import utils

PROFILE_DIR = os.path.join(utils.PROFILE_DIR, "pyinstrument")
os.makedirs(PROFILE_DIR, exist_ok=True)


class PyInstrumentProfiler(utils.Profiler):
    def __init__(self, name: str, print_to_console: bool = False):
        super().__init__(name, print_to_console)
        self.profiler = pyinstrument.Profiler()

    def __enter__(self):
        self.profiler.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop()
        process(self.profiler, self.name, self.print_to_console)


def process(profiler: pyinstrument.Profiler, name: str, print_to_console: bool = False):
    path = os.path.join(PROFILE_DIR, f"{name}")
    if print_to_console:
        print(profiler.output_text(unicode=True, color=True))

    with open(f"{path}.txt", "w", encoding="utf-8") as file:
        file.write(profiler.output_text(unicode=True, color=False))

    with open(f"{path}.html", "w", encoding="utf-8") as file:
        file.write(profiler.output_html())
