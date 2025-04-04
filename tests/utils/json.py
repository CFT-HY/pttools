"""Utilities for processing JSON data"""

import abc
import logging
import os
import os.path
import typing as tp

import numpy as np
import orjson

import pttools.type_hints as th
from tests.utils.assertions import assert_allclose

logger = logging.getLogger(__name__)


class JsonTestCase(abc.ABC):
    """Base class for tests that compare to JSON data"""
    REF_DATA_PATH: str
    data: tp.Dict[str, th.FloatOrArr]
    ref_data: tp.Dict[str, th.FloatOrArr]

    EXPECT_MISSING_DATA: bool = False
    SAVE_NEW_DATA: bool = False

    def assert_json(self, data: th.FloatOrArr, key: str, rtol: float = 1e-7, atol: float = 0, allow_save: bool = True):
        if isinstance(data, np.ndarray):
            if data.size == 1:
                data = data.item()
            if data.ndim == 2 and data.shape[1] == 1:
                data = data.T
        if allow_save:
            self.data[key] = data
        if key in self.ref_data:
            ref_data = self.ref_data[key]
            assert_allclose(data, ref_data, rtol=rtol, atol=atol)
        elif self.EXPECT_MISSING_DATA:
            logger.warning(f"Reference data missing in %s: %s", type(self).__name__, key)
        else:
            raise KeyError(f"Reference data missing in {type(self).__name__}: {key}")

    @classmethod
    # pylint: disable=invalid-name, unused-argument
    def setUpClass(cls, *args, **kwargs):
        cls.data = {}
        if os.path.isfile(cls.REF_DATA_PATH):
            with open(cls.REF_DATA_PATH, "rb") as file:
                # pylint: disable=no-member
                cls.ref_data = orjson.loads(file.read())
        else:
            logger.warning("Reference data file for not found. Starting with a blank file.")
            cls.ref_data = {}

    @classmethod
    # pylint: disable=invalid-name
    def tearDownClass(cls):
        if cls.SAVE_NEW_DATA and cls.data:
            # pylint: disable=no-member
            json = orjson.dumps(
                cls.data,
                option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
            )
            os.makedirs(os.path.dirname(cls.REF_DATA_PATH), exist_ok=True)
            with open(cls.REF_DATA_PATH, "wb") as file:
                file.write(json)
