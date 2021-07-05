"""Performance profiling script

When implementing new tests, the functions should be called at least once to JIT-compile them before profiling
"""

import abc
import unittest

import numpy as np

import pttools.ssmtools as ssm
from pttools import speedup
from tests.paper.test_pow_specs import pow_specs
from . import utils_cprofile
from . import utils_pyinstrument
from . import utils_yappi


class TestProfile(abc.ABC, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not speedup.NUMBA_DISABLE_JIT:
            cls.setup_numba()

    @classmethod
    @abc.abstractmethod
    def setup_numba(cls):
        """Run the command to be profiled before profiling to ensure
        that it's already fully Numba-jitted when profiled."""
        pass


class TestProfileGW(TestProfile):
    name = "gw"
    z = np.logspace(0, 2, 100)
    params = [0.1, 0.1]

    @classmethod
    def setup_numba(cls):
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


class TestProfilePowSpecs(TestProfile):
    name = "pow_specs"

    @classmethod
    def setup_numba(cls):
        pow_specs()

    @classmethod
    def test_profile_pow_specs_cprofile(cls):
        with utils_cprofile.CProfiler(cls.name):
            pow_specs()

    @classmethod
    def test_profile_pow_specs_pyinstrument(cls):
        with utils_pyinstrument.PyInstrumentProfiler(cls.name):
            pow_specs()

    @classmethod
    def test_profile_gw_yappi(cls):
        with utils_yappi.YappiProfiler(cls.name):
            pow_specs()


if __name__ == "__main__":
    unittest.main()
