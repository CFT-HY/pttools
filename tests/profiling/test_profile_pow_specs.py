import os
import unittest

import tests.paper.ssm_paper_utils as spu
from .test_profile import TestProfile
from . import utils_cprofile
from . import utils_pyinstrument
from . import utils_yappi


def pow_specs():
    spu.do_all_plot_ps_compare_nuc('final3', None)


class TestProfilePowSpecs(TestProfile):
    name = "pow_specs"

    @classmethod
    def setUpClass(cls) -> None:
        if os.getenv("GITHUB_ACTIONS", default=False):
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
    def test_profile_pow_specs_pyinstrument(cls):
        with utils_pyinstrument.PyInstrumentProfiler(cls.name):
            pow_specs()

    @classmethod
    def test_profile_pow_specs_yappi(cls):
        with utils_yappi.YappiProfiler(cls.name):
            pow_specs()


if __name__ == "__main__":
    unittest.main()
