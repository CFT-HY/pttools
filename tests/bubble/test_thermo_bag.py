"""Compare thermodynamic quantities of the bag model to the values in references."""

from abc import ABC
import unittest

import numpy as np

from pttools.bubble import (
    CS2_BAG_SCALAR_PTR,
    DEFAULT_FLUID_INTEGRATE_METHOD,
    DF_DTAU_PTR_BAG,
    thermo_bag,
)
from pttools.type_hints import FloatArr1D
from tests.bubble.ref import RefBag, Reference, RefHindmarshHijazi, RefLectureNotes
from tests.utils.test_assertions import assert_allclose


class ThermoBagTest(Reference, ABC):
    """Compare thermodynamic results of the bag model specific functions to the values in a reference."""

    RTOL_KAPPA: float = 6.7e-3
    RTOL_KE_FRAC: float = 6.9e-3

    def test_kappa(self):
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i] = thermo_bag.get_kappa_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=self.RTOL_KAPPA)

    def test_kappa_de(self):
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i], _ = thermo_bag.get_kappa_de_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=self.RTOL_KAPPA)

    def test_kappa_dq(self):
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i], _ = thermo_bag.get_kappa_dq_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=self.RTOL_KAPPA)

    def test_ke_de_frac_bag(self):
        ke_fracs = np.zeros_like(self.KAPPA_REF)
        de_fracs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ke_fracs[i], de_fracs[i] = thermo_bag.get_ke_de_frac_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(ke_fracs, self.BVA_KE_FRAC_REF, rtol=self.RTOL_KE_FRAC)

    def test_ke_frac_bag(self):
        ke_fracs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ke_fracs[i] = thermo_bag.get_ke_frac_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(ke_fracs, self.BVA_KE_FRAC_REF, rtol=self.RTOL_KE_FRAC)

    def test_ke_frac_new_bag(self):
        ke_fracs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ke_fracs[i] = thermo_bag.get_ke_frac_new_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(ke_fracs, self.BVA_KE_FRAC_REF, rtol=self.RTOL_KE_FRAC)


class ThermoBagTestBag(RefBag, ThermoBagTest, unittest.TestCase):
    r"""Compare bag model thermodynamic functions to old PTtools results."""

    RTOL_KAPPA = 0.0126
    RTOL_KE_FRAC = 0.0126


class ThermoBagTestHindmarshHijazi(RefHindmarshHijazi, ThermoBagTest, unittest.TestCase):
    r"""Compare bag model thermodynamic functions to the values in :gw_pt_ssm:`\ `."""


class ThermoBagTestLectureNotes(RefLectureNotes, ThermoBagTest, unittest.TestCase):
    def test_ubarf2(self):
        ubarfs: FloatArr1D = np.zeros_like(self.UBARF_REF)
        for i in range(self.ALPHA_NS.size):
            ubarfs[i] = np.sqrt(thermo_bag.get_ubarf2_bag(
                v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i],
                df_dtau_ptr=DF_DTAU_PTR_BAG,
                ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_ptr=CS2_BAG_SCALAR_PTR))
        assert_allclose(ubarfs, self.UBARF_REF, rtol=2.7e-3)

    def test_ubarf2_new_bag(self):
        ubarfs: FloatArr1D = np.zeros_like(self.UBARF_REF)
        for i in range(self.ALPHA_NS.size):
            ubarfs[i] = np.sqrt(thermo_bag.get_ubarf2_new_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i]))
        assert_allclose(ubarfs, self.UBARF_REF, rtol=5.0e-2)
