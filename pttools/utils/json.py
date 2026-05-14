"""Utilities for exporting data as JSON"""

import typing as tp

from orjson import OPT_APPEND_NEWLINE, OPT_INDENT_2, OPT_SERIALIZE_NUMPY, dumps


def export_json(data: dict[str, tp.Any], path: str | None = None, pretty: bool = True) -> bytes:
    """Export a dictionary as a JSON string"""
    # Pylint doesn't understand orjson
    # pylint: disable=no-member
    json_str = dumps(
        data,
        option=OPT_SERIALIZE_NUMPY | OPT_INDENT_2 | OPT_APPEND_NEWLINE if pretty else OPT_SERIALIZE_NUMPY
    )
    if path is not None:
        with open(path, "wb") as file:
            file.write(json_str)
    return json_str
