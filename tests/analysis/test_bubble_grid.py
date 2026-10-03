"""Test the bubble grid analysis utilities."""

import typing as tp
import unittest

import numpy as np

from pttools.analysis.bubble_grid import BubbleGridVWAlpha
from pttools.models.bag import BagModel
from tests.utils.mark import uses_multiprocessing


class BubbleGridTest(unittest.TestCase):
    """Test the bubble grid of wall speeds and transition strengths."""

    grid: BubbleGridVWAlpha

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        arr = np.linspace(0.1, 0.9, 3)
        cls.grid = BubbleGridVWAlpha(
            model=BagModel(a_s=1.1, a_b=1, V_s=1),
            v_walls=arr,
            alpha_ns=arr
        )

    @uses_multiprocessing
    def test_props(self) -> None:
        """Test that the grid properties provide numerical arrays."""
        arrs = [
            self.grid.kappa(),
            self.grid.numerical_error(),
            self.grid.omega(),
            self.grid.solver_failed(),
            self.grid.unphysical_alpha_plus(),
            self.grid.negative_net_entropy_change()
        ]
        for arr in arrs:
            if arr.dtype == object:
                raise TypeError(f"Array is of object dtype: {arr.dtype}")
