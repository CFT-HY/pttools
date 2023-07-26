# For debugging Numba issues
# import os
# os.environ["NUMBA_DEBUG"] = "1"

import unittest

import numpy as np

from pttools.bubble import quantities
from tests.utils.test_assertions import assert_allclose
# For debugging Numba issues
# from pttools.logging import setup_logging
# setup_logging(silence_spam=False)


class ThermoBagTest:
    """Compare thermodynamic quantities of the bag model to the values in articles."""
    ALPHA_NS: np.ndarray
    V_WALLS: np.ndarray

    KAPPA_REF: np.ndarray
    KE_FRAC_REF: np.ndarray

    def test_kappa(self):
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i] = quantities.get_kappa_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=6.7e-3)

    def test_kappa_de(self):
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i], _ = quantities.get_kappa_de_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=6.7e-3)

    def test_kappa_dq(self):
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i], _ = quantities.get_kappa_dq_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=6.7e-3)

    def test_ke_de_frac_bag(self):
        ke_fracs = np.zeros_like(self.KAPPA_REF)
        de_fracs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ke_fracs[i], de_fracs[i] = quantities.get_ke_de_frac_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(ke_fracs, self.KE_FRAC_REF, rtol=6.9e-3)

    def test_ke_frac_bag(self):
        ke_fracs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ke_fracs[i] = quantities.get_ke_frac_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(ke_fracs, self.KE_FRAC_REF, rtol=6.9e-3)

    def test_ke_frac_new_bag(self):
        ke_fracs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ke_fracs[i] = quantities.get_ke_frac_new_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(ke_fracs, self.KE_FRAC_REF, rtol=6.9e-3)


class ThermoBagTestLectureNotes(ThermoBagTest, unittest.TestCase):
    # Input parameters
    ALPHA_NS = np.array([0.1, 0.1, 0.1])
    V_WALLS = np.array([0.4, 0.7, 0.8])
    # Reference values
    KAPPA_REF = np.array([0.189, 0.452, 0.235])
    OMEGA_REF = np.array([0.815, 0.559, 0.769])
    KE_FRAC_REF = np.array([0.0172, 0.0411, 0.0213])
    UBARFS_REF = np.array([0.119, 0.184, 0.133])

    def test_ubarf2(self):
        ubarfs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ubarfs[i] = np.sqrt(quantities.get_ubarf2_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i]))
        assert_allclose(ubarfs, self.UBARFS_REF, rtol=2.7e-3)

    def test_ubarf2_new_bag(self):
        ubarfs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ubarfs[i] = np.sqrt(quantities.get_ubarf2_new_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i]))
        assert_allclose(ubarfs, self.UBARFS_REF, rtol=5.0e-2)


class ThermoBagTestHindmarshHijazi(ThermoBagTest, unittest.TestCase):
    # Input parameters
    ALPHA_NS = np.array([0.578, 0.151, 0.091])
    V_WALLS = np.array([0.5, 0.7, 0.77])
    # Reference values
    KAPPA_REF = np.array([0.610, 0.522, 0.264])
    OMEGA_REF = np.array([0.395, 0.491, 0.744])
    KE_FRAC_REF = np.array([0.223, 0.0684, 0.022])
