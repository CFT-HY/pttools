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
        """Initialize the pyinstrument profiler.

        :param name: name of the profile, used for the output file names
        :param print_to_console: whether to also print the results to the console
        """
        super().__init__(name, print_to_console)
        self.profiler: pyinstrument.Profiler = pyinstrument.Profiler()

    def __enter__(self) -> None:
        """Start the pyinstrument profiler."""
        self.profiler.start()

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None) -> None:
        """Stop the pyinstrument profiler and save its results."""
        self.profiler.stop()
        process(self.profiler, self.name, self.print_to_console)


def process(profiler: pyinstrument.Profiler, name: str, print_to_console: bool = False) -> None:
    """Save pyinstrument results as text and HTML."""
    path = PROFILE_DIR / name
    if print_to_console:
        print(profiler.output_text(unicode=True, color=True))

    path.with_name(f"{path.name}.txt").write_text(profiler.output_text(unicode=True, color=False), encoding="utf-8")
    path.with_name(f"{path.name}.html").write_text(profiler.output_html(), encoding="utf-8")
