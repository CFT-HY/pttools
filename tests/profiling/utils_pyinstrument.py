"""Wrapper for the pyinstrument profiler."""

from pathlib import Path
import types

import pyinstrument

from tests.profiling import utils

PROFILE_DIR: Path = utils.PROFILE_DIR / "pyinstrument"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


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
    path = PROFILE_DIR / name
    if print_to_console:
        print(profiler.output_text(unicode=True, color=True))

    path.with_name(f"{path.name}.txt").write_text(profiler.output_text(unicode=True, color=False), encoding="utf-8")
    path.with_name(f"{path.name}.html").write_text(profiler.output_html(), encoding="utf-8")
