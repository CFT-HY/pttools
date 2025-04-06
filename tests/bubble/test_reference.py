"""Test the generation of the fluid reference"""

import os
import unittest

from pttools.bubble import fluid_reference
from pttools import speedup
from tests.utils.const import TEST_DATA_PATH


class ReferenceTest(unittest.TestCase):
    """Test the generation of the fluid reference"""

    @unittest.skipIf(speedup.GITHUB_ACTIONS and speedup.IS_WINDOWS, reason="GitHub Actions Windows runners are slow")
    def test_generation(self):
        path = os.path.join(TEST_DATA_PATH, "fluid_reference_indexing.hdf5")
        if os.path.exists(path):
            os.remove(path)
        fluid_reference.FluidReference(
            path=path,
            n_v_wall=5, n_alpha_n=6)

    # def test_nearest(self):
    #     ref = fluid_reference.ref()
    #     v_walls = np.linspace(0.1, 0.9, 3)
    #     alpha_ns = v_walls
    #     for v_wall in v_walls:
    #         for alpha_n in alpha_ns:
    #             ref_grid = ref.get(v_wall, alpha_n, SolutionType.SUB_DEF)
    #             ref_nearest = ref.get(v_wall, alpha_n)
    #             if np.any(np.isnan(ref_grid)):
    #                 continue
    #
    #             assert_allclose(ref_nearest, ref_grid)
