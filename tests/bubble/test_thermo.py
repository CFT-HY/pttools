"""Unit tests for thermodynamic functions."""

from abc import ABC
import unittest

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.bubble.thermo import e_bar, ubarf2, w_bar
from pttools.models.const_cs import ConstCSModel
from tests.bubble.ref import RefBag, Reference, RefHindmarshHijazi, RefLectureNotes
from tests.utils.test_assertions import assert_allclose


class ThermoTest(Reference, ABC):
    """Unit tests for thermodynamic functions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bubbles = [
            Bubble(cls.MODEL, v_wall=v_wall, alpha_n=alpha_n)
            for v_wall, alpha_n in zip(cls.V_WALLS, cls.ALPHA_NS, strict=True)
        ]

    def test_ebar(self):
        assert_allclose(
            [e_bar(model=bubble.model, wn=bubble.wn) for bubble in self.bubbles],
            [bubble.en for bubble in self.bubbles]
        )

    def test_wbar(self):
        """If there is no bubble, then wbar=wn."""
        assert_allclose(
            [
                w_bar(w=np.ones_like(bubble.w) * bubble.wn, xi=bubble.xi, v_wall=bubble.v_wall)
                for bubble in self.bubbles
            ],
            [bubble.wn for bubble in self.bubbles]
        )

    def test_kappa(self):
        assert_allclose([bubble.kappa for bubble in self.bubbles], self.KAPPA_REF, rtol=1.5e-2)

    def test_kappa_omega(self):
        assert_allclose([bubble.kappa + bubble.omega for bubble in self.bubbles], 1, rtol=1.8e-2)

    def test_kappa_omega_ref(self):
        """Ensure that there are no typos in the reference data."""
        assert_allclose(self.KAPPA_REF + self.OMEGA_REF, 1, 1.8e-2)

    def test_bva_ke_frac(self):
        assert_allclose([bubble.kinetic_energy_fraction for bubble in self.bubbles], self.BVA_KE_FRAC_REF, rtol=1.5e-2)

    def test_omega(self):
        assert_allclose([bubble.omega for bubble in self.bubbles], self.OMEGA_REF, rtol=1.3e-2)


class ThermoTestHindmarshHijazi(RefHindmarshHijazi, ThermoTest, unittest.TestCase):
    pass


class ThermoTestLectureNotes(RefLectureNotes, ThermoTest, unittest.TestCase):
    def test_ubarf(self):
        assert_allclose(
            [np.sqrt(ubarf2(
                v=bubble.v, w=bubble.w, xi=bubble.xi,
                v_wall=bubble.v_wall, ek_bva=bubble.kinetic_energy_density, w_bar=bubble.wn
            )) for bubble in self.bubbles],
            self.UBARF_REF, rtol=6.8e-3
        )
        assert_allclose([np.sqrt(bubble.ubarf2) for bubble in self.bubbles], self.UBARF_REF, rtol=0.039)


class ThermoTestBag(RefBag, ThermoTest, unittest.TestCase):
    """Test that the bag results have not changed due to code changes."""


class ThermoTestConstCS(ThermoTest, unittest.TestCase):
    """Test that the results have not changed due to code changes.

    Reference data has been generated with PTtools.
    """

    MODEL = ConstCSModel(css2=1/3-0.01, csb2=1/3, a_s=1.5, a_b=1, V_s=1)

    ALPHA_NS = np.array(np.repeat([0.15, 0.2, 0.3], 3))
    V_WALLS = np.array(np.tile([0.3, 0.7, 0.8], 3))

    KAPPA_REF = np.array([
        1.75263237e-01, 5.29811721e-01, 3.60041399e-01,
        2.18903275e-01, 5.74135884e-01, 4.66461842e-01,
        2.93811401e-01, 6.33146311e-01, 5.83362278e-01
    ])
    OMEGA_REF = np.array([
        8.24598271e-01, 4.77202010e-01, 6.57197385e-01,
        7.80795161e-01, 4.31561444e-01, 5.47128825e-01,
        7.05720787e-01, 3.71190282e-01, 4.26371978e-01
    ])
    BVA_KE_FRAC_REF = np.array([
        2.24107249e-02, 6.85518085e-02, 4.69331415e-02,
        3.57922063e-02, 9.49107619e-02, 7.75841920e-02,
        6.65943334e-02, 1.44939257e-01, 1.34176160e-01
    ])
