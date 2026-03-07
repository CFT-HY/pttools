"""Profile the power spectrum calculation of the paper"""

import logging
import unittest

import tests.paper.ssm_paper_utils as spu
from pttools.speedup.numba_wrapper import \
    NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION, \
    NUMBA_SEGFAULTING_PROFILERS
from pttools.utils.system import IS_GITHUB_ACTIONS
from tests.profiling.test_profile import TestProfile
from tests.profiling import utils_cprofile
from tests.profiling import utils_pyinstrument
from tests.profiling import utils_yappi

logger = logging.getLogger(__name__)


def pow_specs():
    spu.do_all_plot_ps_compare_nuc('final3', None)


class TestProfilePowSpecs(TestProfile):
    """Profile the power spectrum calculation of the paper"""
    name = "pow_specs"

    @classmethod
    def setUpClass(cls) -> None:
        if IS_GITHUB_ACTIONS:
            raise unittest.SkipTest("This test would take too long on GitHub Actions")
        super().setUpClass()

    @classmethod
    def setup_numba(cls):
        pow_specs()

    @classmethod
    def test_profile_pow_specs_cprofile(cls):
        with utils_cprofile.CProfiler(cls.name):
            pow_specs()

    @classmethod
    @unittest.skipIf(
        NUMBA_SEGFAULTING_PROFILERS,
        "Pyinstrument may segfault with old Numba versions")
    def test_profile_pow_specs_pyinstrument(cls):
        try:
            with utils_pyinstrument.PyInstrumentProfiler(cls.name):
                pow_specs()
        except (AssertionError, UnboundLocalError) as e:
            logger.exception("Pyinstrument crashed", exc_info=e)
            if not NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION:
                raise e

    @classmethod
    def test_profile_pow_specs_yappi(cls):
        with utils_yappi.YappiProfiler(cls.name):
            pow_specs()


if __name__ == "__main__":
    unittest.main()
