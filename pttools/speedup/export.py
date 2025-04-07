"""Utlities for exporting data"""

import typing as tp

import orjson


def export_json(data: tp.Dict[str, any], path: str = None) -> bytes:
    # Pylint doesn't understand orjson
    # pylint: disable=no-member
    json_str = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
    if path is not None:
        with open(path, "wb") as file:
            file.write(json_str)
    return json_str
