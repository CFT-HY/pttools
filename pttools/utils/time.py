"""Utilities for time and dates."""

from collections.abc import Callable
import datetime
import functools
import inspect
import logging
import math
from pathlib import Path
import time

__all__ = [
    "format_duration",
    "log_duration",
    "now",
]


def format_duration(duration: float | datetime.timedelta) -> str:
    """Format a duration in the form "1 h 2 min 3.123 s".

    The leading hours and minutes are omitted when they are zero, e.g. "2 min 3.123 s" or "3.123 s",
    and the seconds are given with three decimals.
    The duration is rounded to milliseconds before splitting it into the units,
    so that e.g. 59.9996 s is formatted as "1 min 0.000 s" instead of "60.000 s".
    Negative durations are formatted with a leading minus sign, e.g. "-2 min 3.123 s".

    :param duration: the duration in seconds or as a :py:class:`datetime.timedelta`
    :return: the formatted duration
    """
    seconds = duration.total_seconds() if isinstance(duration, datetime.timedelta) else float(duration)
    if not math.isfinite(seconds):
        return f"{seconds} s"
    # The milliseconds are rounded half up instead of to the nearest even number as by round().
    total_milliseconds = math.floor(abs(seconds) * 1000 + 0.5)
    sign = "-" if seconds < 0 and total_milliseconds else ""
    hours, milliseconds = divmod(total_milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} h")
    if hours or minutes:
        parts.append(f"{minutes} min")
    parts.append(f"{milliseconds // 1000}.{milliseconds % 1000:03d} s")
    return sign + " ".join(parts)


def log_duration[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that logs the time spent by each call of the decorated function.

    The time is logged at the ``INFO`` level with the logger of the module of the function,
    together with the qualified name of the function and the name of the file in which it's defined.
    The time is logged also when the function raises an exception.

    :param func: the function to be decorated
    :return: the decorated function
    """
    logger = logging.getLogger(func.__module__)
    name: str = getattr(func, "__qualname__", repr(func))
    try:
        file_name = Path(inspect.getfile(func)).name
    except TypeError:
        # Built-in functions are not defined in a file.
        file_name = "unknown file"

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        success = False
        try:
            ret = func(*args, **kwargs)
            success = True
            return ret
        finally:
            duration = format_duration(time.perf_counter() - start_time)
            logger.info("%s in %s %s %s.", name, file_name, "took" if success else "failed after", duration)

    return wrapper


def now(_obj: object = None) -> datetime.datetime:
    """Current local time, including the time zone.

    :param _obj: ignored. This makes the function usable as the getter of a :py:class:`pttools.utils.fields.Field`.
    :return: the current time with the local time zone
    """
    return datetime.datetime.now().astimezone()
