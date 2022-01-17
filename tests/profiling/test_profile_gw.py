import unittest

import numpy as np

import pttools.ssmtools as ssm
from pttools import speedup
from .test_profile import TestProfile
from . import utils_cprofile
from . import utils_pyinstrument
from . import utils_yappi


class TestProfileGW(TestProfile):
    name = "gw"
    z = np.logspace(0, 2, 100)
    params = (0.1, 0.1)

    @classmethod
    def setup_numba(cls):
        ssm.power_gw_scaled(cls.z, cls.params)

    @classmethod
    def test_profile_gw_cprofile(cls):
        with utils_cprofile.CProfiler(cls.name):
            ssm.power_gw_scaled(cls.z, cls.params)

    @classmethod
    @unittest.skipIf(
        speedup.NUMBA_SEGFAULTING_PROFILERS,
        "Pyinstrument may segfault with old Numba versions")
    def test_profile_gw_pyinstrument(cls):
        """Pyinstrument is a sampling profiler, and therefore repeating gives more accurate results."""
        with utils_pyinstrument.PyInstrumentProfiler(cls.name):
            for _ in range(100):
                ssm.power_gw_scaled(cls.z, cls.params)

    @classmethod
    def test_profile_gw_yappi(cls):
        with utils_yappi.YappiProfiler(cls.name):
            ssm.power_gw_scaled(cls.z, cls.params)


if __name__ == "__main__":
    unittest.main()
