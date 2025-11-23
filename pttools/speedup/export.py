"""Utilities for exporting data"""

import typing as tp

import orjson


def export_json(data: dict[str, tp.Any], path: str | None = None) -> bytes:
    """Export a dictionary as a JSON string"""
    # Pylint doesn't understand orjson
    # pylint: disable=no-member
    json_str = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
    if path is not None:
        with open(path, "wb") as file:
            file.write(json_str)
    return json_str
