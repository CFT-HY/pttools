"""Profile GW performance"""

import logging
import unittest

import numpy as np

from pttools import ssm
from pttools import speedup
from tests.profiling.test_profile import TestProfile
from tests.profiling import utils_cprofile
from tests.profiling import utils_pyinstrument
from tests.profiling import utils_yappi
from tests.utils.mark import skip_slow

logger = logging.getLogger(__name__)


class TestProfileGW(TestProfile):
    """Profile GW performance"""
    name = "gw"
    z = np.logspace(0, 2, 100)
    params = (0.1, 0.1)

    @classmethod
    def setup_numba(cls):
        ssm.power_gw_bag(cls.z, cls.params)

    @classmethod
    @skip_slow
    def test_profile_gw_cprofile(cls):
        with utils_cprofile.CProfiler(cls.name):
            ssm.power_gw_bag(cls.z, cls.params)

    @classmethod
    @skip_slow
    @unittest.skipIf(
        speedup.NUMBA_SEGFAULTING_PROFILERS,
        "Pyinstrument may segfault with old Numba versions")
    def test_profile_gw_pyinstrument(cls):
        """Pyinstrument is a sampling profiler, and therefore repeating gives more accurate results."""
        try:
            with utils_pyinstrument.PyInstrumentProfiler(cls.name):
                for _ in range(100):
                    ssm.power_gw_bag(cls.z, cls.params)
        except (AssertionError, UnboundLocalError) as e:
            logger.exception("Pyinstrument crashed", exc_info=e)
            if not speedup.NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION:
                raise e

    @classmethod
    @skip_slow
    def test_profile_gw_yappi(cls):
        with utils_yappi.YappiProfiler(cls.name):
            ssm.power_gw_bag(cls.z, cls.params)


if __name__ == "__main__":
    unittest.main()
