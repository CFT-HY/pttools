"""Profile the power spectrum calculation of the paper."""

import logging
import typing as tp
import unittest

from pttools.speedup.numba_wrapper import NUMBA_PYINSTRUMENT_INCOMPATIBLE_PYTHON_VERSION, NUMBA_SEGFAULTING_PROFILERS
from pttools.utils.system import IS_GITHUB_ACTIONS
import tests.paper.ssm_paper_utils as spu
from tests.profiling import utils_cprofile, utils_pyinstrument, utils_yappi
from tests.profiling.test_profile import TestProfile
from tests.utils.mark import skip_slow

logger: logging.Logger = logging.getLogger(__name__)


def pow_specs() -> None:
    """Compute the power spectra of the paper with simultaneous and exponential nucleation."""
    spu.do_all_plot_ps_compare_nuc('final3', None)


class TestProfilePowSpecs(TestProfile):
    """Profile the power spectrum calculation of the paper."""

    NAME = "pow_specs"

    @classmethod
    def setUpClass(cls) -> None:
        """Skip the tests on GitHub Actions, as they would take too long, and otherwise JIT-compile the code."""
        if IS_GITHUB_ACTIONS:
            raise unittest.SkipTest("This test would take too long on GitHub Actions")
        super().setUpClass()

    @classmethod
    @tp.override
    def setup_numba(cls) -> None:
        pow_specs()

    @classmethod
    @skip_slow
    def test_profile_pow_specs_cprofile(cls) -> None:
        """Profile the power spectrum calculation of the paper with cProfile."""
        with utils_cprofile.CProfiler(cls.NAME):
            pow_specs()

    @classmethod
    @skip_slow
    @unittest.skipIf(
        NUMBA_SEGFAULTING_PROFILERS,
        "Pyinstrument may segfault with old Numba versions")
    def test_profile_pow_specs_pyinstrument(cls) -> None:
        """Profile the power spectrum calculation of the paper with pyinstrument."""
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
        """Profile the power spectrum calculation of the paper with YAPPI."""
        with utils_yappi.YappiProfiler(cls.NAME):
            pow_specs()


if __name__ == "__main__":
    unittest.main()
