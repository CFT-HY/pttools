import cProfile
import io
import os
import pstats
import sys
import typing as tp

from . import utils

PROFILE_DIR = os.path.join(utils.PROFILE_DIR, "cprofile")
if not os.path.isdir(PROFILE_DIR):
    os.mkdir(PROFILE_DIR)


class CProfiler(utils.Profiler):
    def __init__(self, name: str):
        super().__init__(name)
        self.profiler = cProfile.Profile()

    def __enter__(self):
        self.profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        process(self.name, self.profiler)


def process(name: str, profile: cProfile.Profile):
    path = os.path.join(PROFILE_DIR, f"{name}")
    profile.dump_stats(f"{path}.pstat")

    save_sorted(profile, path, "time", True)
    save_sorted(profile, path, "cumulative")
    save_sorted(profile, path, "pcalls")


def save_sorted(
        profile: cProfile.Profile,
        path: str,
        sort: tp.Union["pstats.SortKey", str],
        print_to_console: bool = False):
    # Save to file
    stream = io.StringIO()
    ps = pstats.Stats(profile, stream=stream).sort_stats(sort)
    ps.print_stats()
    text = stream.getvalue()
    if print_to_console:
        print(text)

    # pstats.SortKey was introduced in Python 3.7
    sort_name = sort.value if (sys.version_info >= (3, 7) and sort is pstats.SortKey) else sort
    path_labeled = f"{path}_{sort_name}"
    with open(f"{path_labeled}.txt", "w") as file:
        file.write(text)
    save_filtered(text, f"{path_labeled}_numba.txt", os.path.join("site-packages", "numba"))
    save_filtered(text, f"{path_labeled}_all.txt", "site-packages")


def save_filtered(text: str, path: str, filter_text: str):
    lines = text.splitlines(keepends=True)
    with open(path, "w") as file:
        for line in lines:
            if filter_text not in line:
                file.write(line)
