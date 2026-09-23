"""Wrapper for the pyinstrument profiler."""

import os
import types

import pyinstrument

from tests.profiling import utils

PROFILE_DIR: str = os.path.join(utils.PROFILE_DIR, "pyinstrument")
os.makedirs(PROFILE_DIR, exist_ok=True)


class PyInstrumentProfiler(utils.Profiler):
    """Wrapper for the pyinstrument profiler."""

    def __init__(self, name: str, print_to_console: bool = False) -> None:
        super().__init__(name, print_to_console)
        self.profiler: pyinstrument.Profiler = pyinstrument.Profiler()

    def __enter__(self) -> None:
        self.profiler.start()

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None) -> None:
        self.profiler.stop()
        process(self.profiler, self.name, self.print_to_console)


def process(profiler: pyinstrument.Profiler, name: str, print_to_console: bool = False) -> None:
    path = os.path.join(PROFILE_DIR, f"{name}")
    if print_to_console:
        print(profiler.output_text(unicode=True, color=True))

    with open(f"{path}.txt", "w", encoding="utf-8") as file:
        file.write(profiler.output_text(unicode=True, color=False))

    with open(f"{path}.html", "w", encoding="utf-8") as file:
        file.write(profiler.output_html())
