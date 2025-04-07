"""Profile the power spectrum calculation of the paper"""

import logging
import unittest

import tests.paper.ssm_paper_utils as spu
from pttools import speedup
from .test_profile import TestProfile
from . import utils_cprofile
from . import utils_pyinstrument
from . import utils_yappi

logger = logging.getLogger(__name__)


def pow_specs():
    spu.do_all_plot_ps_compare_nuc('final3', None)


class TestProfilePowSpecs(TestProfile):
    """Profile the power spectrum calculation of the paper"""
    name = "pow_specs"

    @classmethod
    def setUpClass(cls) -> None:
        if speedup.GITHUB_ACTIONS:
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
        speedup.NUMBA_SEGFAULTING_PROFILERS,
        "Pyinstrument may segfault with old Numba versions")
    def test_profile_pow_specs_pyinstrument(cls):
        try:
            with utils_pyinstrument.PyInstrumentProfiler(cls.name):
                pow_specs()
        except (AssertionError, UnboundLocalError) as e:
            logger.exception("Pyinstrument crashed", exc_info=e)
            if not speedup.NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION:
                raise e

    @classmethod
    def test_profile_pow_specs_yappi(cls):
        with utils_yappi.YappiProfiler(cls.name):
            pow_specs()


if __name__ == "__main__":
    unittest.main()
