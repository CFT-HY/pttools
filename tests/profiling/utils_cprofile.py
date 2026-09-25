"""Wrapper for the cProfile profiler."""

import cProfile
import io
import os
from pathlib import Path
import pstats
import types

from tests.profiling import utils

PROFILE_DIR: Path = utils.PROFILE_DIR / "cprofile"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


class CProfiler(utils.Profiler):
    """Wrapper for the cProfile profiler."""

    def __init__(self, name: str, print_to_console: bool = False) -> None:
        super().__init__(name, print_to_console)
        self.profiler: cProfile.Profile = cProfile.Profile()

    def __enter__(self) -> None:
        self.profiler.enable()

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None) -> None:
        self.profiler.disable()
        process(self.name, self.profiler, self.print_to_console)


def process(name: str, profile: cProfile.Profile, print_to_console: bool = False) -> None:
    """Process and save cProfile results."""
    path = PROFILE_DIR / name
    profile.dump_stats(f"{path}.pstat")

    save_sorted(profile, path, "time", print_to_console)
    save_sorted(profile, path, "cumulative")
    save_sorted(profile, path, "pcalls")


def save_sorted(
        profile: cProfile.Profile,
        path: str | os.PathLike[str],
        sort: pstats.SortKey | str,
        print_to_console: bool = False) -> None:
    """Save sorted cProfile results to file."""
    # Save to file
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats(sort)
    stats.print_stats()
    text = stream.getvalue()
    if print_to_console:
        print(text)

    sort_name = sort.value if isinstance(sort, pstats.SortKey) else sort
    path_labeled = f"{path}_{sort_name}"
    Path(f"{path_labeled}.txt").write_text(text)
    save_filtered(text, f"{path_labeled}_numba.txt", str(Path("site-packages", "numba")))
    save_filtered(text, f"{path_labeled}_all.txt", "site-packages")


def save_filtered(text: str, path: str | os.PathLike[str], filter_text: str) -> None:
    lines = text.splitlines(keepends=True)
    with Path(path).open("w") as file:
        for line in lines:
            if filter_text not in line:
                file.write(line)
