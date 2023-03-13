import os
import unittest

import numpy as np

from pttools.bubble import fluid_reference
from tests.utils.const import TEST_DATA_PATH
from tests.utils.test_assertions import assert_allclose


class ReferenceTest(unittest.TestCase):
    def test_indexing(self):
        path = os.path.join(TEST_DATA_PATH, "fluid_reference_indexing.hdf5")
        if os.path.exists(path):
            os.remove(path)
        fluid_reference.FluidReference(
            path=path,
            n_v_wall=2, n_alpha_n=3)

    def test_nearest(self):
        ref = fluid_reference.ref()
        v_walls = np.linspace(0.1, 0.9, 3)
        alpha_ns = v_walls
        for v_wall in v_walls:
            for alpha_n in alpha_ns:
                ref_grid = ref.get(v_wall, alpha_n, allow_nan=True)
                ref_nearest = ref.get(v_wall, alpha_n)
                if np.any(np.isnan(ref_grid)):
                    continue

                assert_allclose(ref_nearest, ref_grid)
