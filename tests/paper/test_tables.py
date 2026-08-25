"""Test the generation of data tables for the paper"""

import io
import os.path
import typing as tp
import unittest

import numpy as np

from pttools.speedup.options import NUMBA_INTEGRATE
import pttools.type_hints as th
from pttools.utils.decorators import conditional_decorator
from tests.paper import ssm_paper_utils as spu
from tests.utils.const import TEST_DATA_PATH


class TestTables(unittest.TestCase):
    """Test the generation of data tables for the paper"""
    params: th.FloatArr2D
    v2: th.FloatArr2D
    Omgw: th.FloatArr2D
    pfit_sim: th.FloatArr2D
    pfit_exp: th.FloatArr2D

    @classmethod
    def setUpClass(cls) -> None:
        data = np.loadtxt(os.path.join(TEST_DATA_PATH, "data_compare_nuc-final3.txt"))
        cls.params = data[:, 0:2]
        cls.v2 = data[:, 2:4]
        cls.Omgw = data[:, 4:6]
        cls.pfit_sim = data[:, 6:8]
        cls.pfit_exp = data[:, 8:]

    def validate(self, name: str, func: tp.Callable, args):
        path = os.path.join(TEST_DATA_PATH, name)

        # Generate new reference data
        # func(*args, file_name=path)

        buffer = io.StringIO()
        func(*args, file_name=buffer)
        with open(path, "r") as file:
            ref_data = file.read()

        buffer.seek(0)
        self.assertEqual(buffer.read(), ref_data)

    @conditional_decorator(unittest.expectedFailure, NUMBA_INTEGRATE)
    def test_1dh_compare_table(self):
        self.validate(
            "table_1dh_compare.tex",
            spu.make_1dh_compare_table,
            [self.params, self.v2])

    def test_3dh_compare_table(self):
        self.validate(
            "table_3dh_compare.tex",
            spu.make_3dh_compare_table,
            [self.params, self.v2, self.Omgw, self.pfit_sim])

    def test_nuc_compare_table(self):
        self.validate(
            "table_nuc_compare.tex",
            spu.make_nuc_compare_table,
            [self.params, self.v2, self.Omgw, self.pfit_sim, self.pfit_exp]
        )


if __name__ == "__main__":
    unittest.main()
