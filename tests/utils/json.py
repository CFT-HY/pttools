"""Utilities for processing JSON data."""

import abc
import logging
from pathlib import Path
import typing as tp

import numpy as np
import orjson

import pttools.type_hints as th
from pttools.utils.assertions import assert_allclose

logger: logging.Logger = logging.getLogger(__name__)


class JsonTestCase(abc.ABC):
    """Base class for tests that compare to JSON data."""

    REF_DATA_PATH: Path
    data: dict[str, th.FloatOrArr]
    ref_data: dict[str, th.FloatOrArr]

    EXPECT_MISSING_DATA: bool = False
    SAVE_NEW_DATA: bool = False

    def assert_json(
            self, data: th.FloatOrArr, key: str, rtol: float = 1e-7, atol: float = 0, allow_save: bool = True) -> None:
        if isinstance(data, np.ndarray):
            if data.size == 1:
                data = data.item()
            elif data.shape[1:] == (1,):
                data = data.T
        if allow_save:
            self.data[key] = data
        if key in self.ref_data:
            ref_data = self.ref_data[key]
            assert_allclose(data, ref_data, rtol=rtol, atol=atol)
        elif self.EXPECT_MISSING_DATA:
            logger.warning("Reference data missing in %s: %s", type(self).__name__, key)
        else:
            raise KeyError(f"Reference data missing in {type(self).__name__}: {key}")

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        cls.data = {}
        if cls.REF_DATA_PATH.is_file():
            cls.ref_data = orjson.loads(cls.REF_DATA_PATH.read_bytes())
        else:
            logger.warning("Reference data file for not found. Starting with a blank file.")
            cls.ref_data = {}

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.SAVE_NEW_DATA and cls.data:
            json = orjson.dumps(
                cls.data,
                option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
            )
            cls.REF_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            cls.REF_DATA_PATH.write_bytes(json)
