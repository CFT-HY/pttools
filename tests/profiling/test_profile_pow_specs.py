import unittest

from tests.paper.test_pow_specs import pow_specs
from .test_profile import TestProfile
from . import utils_cprofile
from . import utils_pyinstrument
from . import utils_yappi


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
