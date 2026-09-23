"""Profile GW performance."""

import logging
import unittest

import numpy as np

from pttools import speedup, ssm
from tests.profiling import utils_cprofile, utils_pyinstrument, utils_yappi
from tests.profiling.test_profile import TestProfile
from tests.utils.mark import skip_slow

logger: logging.Logger = logging.getLogger(__name__)

#: Change this to e.g. 100 to obtain proper profiling data for pyinstrument.
#: This is set to 1 to speed up the unit testing.
N_ITERATIONS: int = 1


class TestProfileGW(TestProfile):
    """Profile GW performance."""

    NAME = "gw"
    z = np.logspace(0, 2, 100)
    params = (0.1, 0.1)

    @classmethod
    def setup_numba(cls) -> None:
        ssm.power_gw_bag(cls.z, cls.params)

    @classmethod
    @skip_slow
    def test_profile_gw_cprofile(cls) -> None:
        with utils_cprofile.CProfiler(cls.NAME):
            ssm.power_gw_bag(cls.z, cls.params)

    @classmethod
    @skip_slow
    @unittest.skipIf(
        speedup.NUMBA_SEGFAULTING_PROFILERS,
        "Pyinstrument may segfault with old Numba versions")
    def test_profile_gw_pyinstrument(cls) -> None:
        """Pyinstrument is a sampling profiler, and therefore repeating gives more accurate results."""
        try:
            with utils_pyinstrument.PyInstrumentProfiler(cls.NAME):
                for _ in range(N_ITERATIONS):
                    ssm.power_gw_bag(cls.z, cls.params)
        except (AssertionError, UnboundLocalError) as e:
            logger.exception("Pyinstrument crashed", exc_info=e)
            if not speedup.NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION:
                raise e

    @classmethod
    @skip_slow
    def test_profile_gw_yappi(cls) -> None:
        with utils_yappi.YappiProfiler(cls.NAME):
            ssm.power_gw_bag(cls.z, cls.params)


if __name__ == "__main__":
    unittest.main()
