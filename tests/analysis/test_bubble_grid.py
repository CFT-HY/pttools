import unittest

import numpy as np

from pttools.analysis.bubble_grid import BubbleGridVWAlpha
from pttools.models.bag import BagModel


class BubbleGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        arr = np.linspace(0.1, 0.9, 3)
        cls.grid = BubbleGridVWAlpha(
            model=BagModel(a_s=1.1, a_b=1, V_s=1),
            v_walls=arr,
            alpha_ns=arr
        )

    def test_props(self):
        arrs = [
            self.grid.kappa(),
            self.grid.numerical_error(),
            self.grid.omega(),
            self.grid.solver_failed(),
            self.grid.unphysical_alpha_plus(),
            self.grid.unphysical_entropy()
        ]
        for arr in arrs:
            if arr.dtype == np.object:
                raise TypeError(f"Array is of object dtype: {arr.dtype}")
