"""Profile the power spectrum calculation of the paper."""

import logging
import unittest

from pttools.speedup.numba_wrapper import NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION, NUMBA_SEGFAULTING_PROFILERS
from pttools.utils.system import IS_GITHUB_ACTIONS
import tests.paper.ssm_paper_utils as spu
from tests.profiling import utils_cprofile, utils_pyinstrument, utils_yappi
from tests.profiling.test_profile import TestProfile
from tests.utils.mark import skip_slow

logger: logging.Logger = logging.getLogger(__name__)


def pow_specs() -> None:
    spu.do_all_plot_ps_compare_nuc('final3', None)


class TestProfilePowSpecs(TestProfile):
    """Profile the power spectrum calculation of the paper."""

    NAME = "pow_specs"

    @classmethod
    def setUpClass(cls) -> None:
        if IS_GITHUB_ACTIONS:
            raise unittest.SkipTest("This test would take too long on GitHub Actions")
        super().setUpClass()

    @classmethod
    def setup_numba(cls) -> None:
        pow_specs()

    @classmethod
    @skip_slow
    def test_profile_pow_specs_cprofile(cls) -> None:
        with utils_cprofile.CProfiler(cls.NAME):
            pow_specs()

    @classmethod
    @skip_slow
    @unittest.skipIf(
        NUMBA_SEGFAULTING_PROFILERS,
        "Pyinstrument may segfault with old Numba versions")
    def test_profile_pow_specs_pyinstrument(cls) -> None:
        try:
            with utils_pyinstrument.PyInstrumentProfiler(cls.NAME):
                pow_specs()
        except (AssertionError, UnboundLocalError) as e:
            logger.exception("Pyinstrument crashed", exc_info=e)
            if not NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION:
                raise e

    @classmethod
    @skip_slow
    def test_profile_pow_specs_yappi(cls) -> None:
        with utils_yappi.YappiProfiler(cls.NAME):
            pow_specs()


if __name__ == "__main__":
    unittest.main()
