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
    "format_time",
    "log_duration",
    "now",
]


def format_duration(duration: float | datetime.timedelta, days: bool = True, seconds: bool = True) -> str:
    """Format a duration in the form "1 h 2 min 3.123 s".

    The leading hours and minutes are omitted when they are zero, e.g. "2 min 3.123 s" or "3.123 s",
    and the seconds are given with three decimals.
    The duration is rounded to milliseconds before splitting it into the units,
    so that e.g. 59.9996 s is formatted as "1 min 0.000 s" instead of "60.000 s".
    Negative durations are formatted with a leading minus sign, e.g. "-2 min 3.123 s".

    The whole days are given separately, e.g. "1 d 2 h 3 min 4.000 s",
    unless ``days=False``, with which they are included in the hours, e.g. "26 h 3 min 4.000 s".
    With ``seconds=False``, the duration is rounded down to whole minutes, and the seconds are omitted,
    e.g. "1 h 2 min" or "0 min".

    :param duration: the duration in seconds or as a :py:class:`datetime.timedelta`
    :param days: whether to give the whole days separately instead of including them in the hours
    :param seconds: whether to include the seconds
    :return: the formatted duration
    """
    total_seconds = duration.total_seconds() if isinstance(duration, datetime.timedelta) else float(duration)
    if not math.isfinite(total_seconds):
        return f"{total_seconds} s" if seconds else f"{total_seconds} min"
    # The milliseconds are rounded half up instead of to the nearest even number as by round().
    total_milliseconds = math.floor(abs(total_seconds) * 1000 + 0.5)
    total_minutes, milliseconds = divmod(total_milliseconds, 60_000)
    if not seconds:
        milliseconds = 0
    total_hours, minutes = divmod(total_minutes, 60)
    n_days, hours = divmod(total_hours, 24) if days else (0, total_hours)
    sign = "-" if total_seconds < 0 and (total_milliseconds if seconds else total_minutes) else ""
    parts: list[str] = []
    if n_days:
        parts.append(f"{n_days} d")
    if n_days or hours:
        parts.append(f"{hours} h")
    if parts or minutes or not seconds:
        parts.append(f"{minutes} min")
    if seconds:
        parts.append(f"{milliseconds // 1000}.{milliseconds % 1000:03d} s")
    return sign + " ".join(parts)


def format_time(value: datetime.datetime) -> str:
    """Format a time in a human-readable format in the local time zone.

    :param value: the time. A time without time zone information is assumed to be in the local time zone.
    :return: the time, e.g. ``Sun 2026-10-04 12:25:06 EEST``
    """
    return value.astimezone().strftime("%a %Y-%m-%d %H:%M:%S %Z")


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
