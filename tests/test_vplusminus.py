import os.path
import unittest

import numpy as np

from pttools import bubble
from test_utils import TEST_DATA_PATH


class TestVPlusMinus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.npts = 500
        cls.alpha_plus_list = [0.0, 0.01, 0.1, 0.3]

    def v_conversion(self, func: callable, ref_path: str, v_first: np.ndarray):
        data = [v_first]
        for i_alpha, alpha in enumerate(self.alpha_plus_list):
            data.append(func(v_first, alpha, 'Detonation'))
            data.append(func(v_first, alpha, 'Deflagration'))

        # Generate new reference data
        # np.savetxt(ref_path, np.array(data).T)

        data_arr = np.array(data).T
        data_ref = np.loadtxt(ref_path)
        np.testing.assert_array_equal(data_arr, data_ref)

    def test_v_plus_minus(self):
        """Compute v_plus from v_minus.
        This generates the same data as plotted by sound-shell-model/paper/python/fig_8l_vplusminus.py.
        """
        v_first = np.linspace(1 / self.npts, 1, self.npts)
        self.v_conversion(bubble.v_plus, os.path.join(TEST_DATA_PATH, "v_plus_minus.txt"), v_first)

    def test_v_minus_plus(self):
        """Compute v_minus from v_plus."""
        # Todo: Test in some other way to avoid "RuntimeWarning: invalid value encountered in sqrt"
        v_first = np.linspace(1/self.npts+0.1, 0.9, self.npts)
        self.v_conversion(bubble.v_minus, os.path.join(TEST_DATA_PATH, "v_minus_plus.txt"), v_first)
