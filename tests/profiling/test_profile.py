"""Performance profiling script

When implementing new tests, the functions should be called at least once to JIT-compile them before profiling
"""

import unittest

import numpy as np

import pttools.ssmtools as ssm
from tests.paper.test_pow_specs import TestPowSpecs
from . import utils_cprofile
from . import utils_pyinstrument
from . import utils_yappi


class TestProfileGW(unittest.TestCase):
    name = "gw"
    z = np.logspace(0, 2, 100)
    params = [0.1, 0.1]

    @classmethod
    def setUpClass(cls) -> None:
        ssm.power_gw_scaled(cls.z, cls.params)

    @classmethod
    def test_profile_gw_cprofile(cls):
        with utils_cprofile.CProfiler(cls.name):
            ssm.power_gw_scaled(cls.z, cls.params)

    @classmethod
    def test_profile_gw_pyinstrument(cls):
        with utils_pyinstrument.PyInstrumentProfiler(cls.name):
            ssm.power_gw_scaled(cls.z, cls.params)

    @classmethod
    def test_profile_gw_yappi(cls):
        with utils_yappi.YappiProfiler(cls.name):
            ssm.power_gw_scaled(cls.z, cls.params)


class TestProfilePowSpecs(unittest.TestCase):
    name = "pow_specs"

    @classmethod
    def setUpClass(cls) -> None:
        TestPowSpecs.test_pow_specs()

    @classmethod
    def test_profile_pow_specs_cprofile(cls):
        TestPowSpecs.test_pow_specs()
        with utils_cprofile.CProfiler(cls.name):
            TestPowSpecs.test_pow_specs()

    @classmethod
    def test_profile_pow_specs_pyinstrument(cls):
        TestPowSpecs.test_pow_specs()
        with utils_pyinstrument.PyInstrumentProfiler(cls.name):
            TestPowSpecs.test_pow_specs()

    @classmethod
    def test_profile_gw_yappi(cls):
        with utils_yappi.YappiProfiler(cls.name):
            TestPowSpecs.test_pow_specs()


if __name__ == "__main__":
    unittest.main()
