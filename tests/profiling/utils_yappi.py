"""Wrapper for the YAPPI profiler"""

import io
import os
import threading
import typing as tp

import yappi

from . import utils

PROFILE_DIR = os.path.join(utils.PROFILE_DIR, "yappi")
os.makedirs(PROFILE_DIR, exist_ok=True)


class YappiProfiler(utils.Profiler):
    """Thread-safe handler for the YAPPI profiler"""
    _lock = threading.Lock()

    @classmethod
    def __enter__(cls):
        if cls._lock.locked() or yappi.is_running():
            raise RuntimeError("Yappi does not support concurrent sessions")
        cls._lock.acquire()
        yappi.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        yappi.stop()
        process(self.name, self.print_to_console)


def process_text_func(stats: yappi.YFuncStats, path: str, print_to_console: bool = False) -> str:
    """Convert YAPPI function stats to str"""
    return process_text(stats, path, print_to_console, columns={
        0: ("ncall", 5),
        1: ("tsub", 8),
        2: ("ttot", 8),
        3: ("tavg", 8),
        4: ("name", 100)
    })


def process_text_thread(stats: yappi.YThreadStats, path: str, print_to_console: bool = False) -> str:
    """Convert YAPPI thread stats to str"""
    return process_text(stats, path, print_to_console, columns={
        0: ("name", 20),
        1: ("id", 5),
        2: ("tid", 15),
        3: ("ttot", 8),
        4: ("scnt", 10)
    })


def process_text(
        stats: tp.Union[yappi.YFuncStats, yappi.YThreadStats],
        path: str = None,
        print_to_console: bool = False,
        columns: tp.Dict[int, tp.Tuple[str, int]] = None) -> str:
    """Convert YAPPI stats to str"""
    stream = io.StringIO()
    kwargs = {"out": stream}
    if columns:
        kwargs["columns"] = columns
    stats.print_all(**kwargs)
    text = stream.getvalue()

    if print_to_console:
        print(text)
    if path:
        with open(path, "w") as file:
            file.write(text)

    return text


def process(name: str, print_to_console: bool = False) -> tp.Tuple[yappi.YFuncStats, yappi.YThreadStats]:
    """Get stats from YAPPI and return them as str"""
    func_stats: yappi.YFuncStats = yappi.get_func_stats()
    thread_stats: yappi.YThreadStats = yappi.get_thread_stats()

    if print_to_console:
        func_stats.print_all()
    path = os.path.join(PROFILE_DIR, f"{name}")
    path_func = f"{path}_functions"
    for fmt, extension in [("YSTAT", "ystat"), ("CALLGRIND", "callgrind"), ("PSTAT", "pstat")]:
        func_stats.save(f"{path_func}.{extension}", type=fmt)

    process_text_func(func_stats, f"{path_func}.txt", print_to_console)
    process_text_thread(thread_stats, f"{path}_threads.txt", print_to_console)

    return func_stats, thread_stats
