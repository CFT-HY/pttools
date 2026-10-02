"""Utilities for time and dates."""

import datetime

__all__ = [
    "now",
]


def now(_obj: object = None) -> datetime.datetime:
    """Current local time, including the time zone.

    :param _obj: ignored. This makes the function usable as the getter of a :py:class:`pttools.utils.fields.Field`.
    :return: the current time with the local time zone
    """
    return datetime.datetime.now().astimezone()
