import typing as tp
import unittest

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.models.model import Model
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel
from tests.utils.test_assertions import assert_allclose


class ThermoTest:
    MODEL: Model = BagModel(a_s=1.1, a_b=1, V_s=1)

    ALPHA_NS: np.ndarray
    V_WALLS: np.ndarray

    KAPPA_REF: np.ndarray
    OMEGA_REF: np.ndarray
    BVA_KE_FRAC_REF: np.ndarray

    bubbles: tp.List[Bubble]

    @classmethod
    # pylint: disable=invalid-name
    def setUpClass(cls) -> None:
        cls.bubbles = [
            Bubble(cls.MODEL, v_wall=v_wall, alpha_n=alpha_n)
            for v_wall, alpha_n in zip(cls.V_WALLS, cls.ALPHA_NS)
        ]

    def test_kappa(self):
        assert_allclose([bubble.kappa for bubble in self.bubbles], self.KAPPA_REF, rtol=1.5e-2)

    def test_kappa_omega(self):
        assert_allclose([bubble.kappa + bubble.omega for bubble in self.bubbles], 1, rtol=1.8e-2)

    def test_kappa_omega_ref(self):
        """Ensure that there are no typos in the reference data"""
        assert_allclose(self.KAPPA_REF + self.OMEGA_REF, 1, 1.8e-2)

    def test_bva_ke_frac(self):
        assert_allclose([bubble.kinetic_energy_fraction for bubble in self.bubbles], self.BVA_KE_FRAC_REF, rtol=1.5e-2)

    def test_omega(self):
        assert_allclose([bubble.omega for bubble in self.bubbles], self.OMEGA_REF, rtol=1.3e-2)


class ThermoTestLectureNotes(ThermoTest, unittest.TestCase):
    # Input parameters
    ALPHA_NS = np.array([0.1, 0.1, 0.1])
    V_WALLS = np.array([0.4, 0.7, 0.8])
    # Reference values
    KAPPA_REF = np.array([0.189, 0.452, 0.235])
    OMEGA_REF = np.array([0.815, 0.559, 0.769])
    BVA_KE_FRAC_REF = np.array([0.0172, 0.0411, 0.0213])
    UBARFS_REF = np.array([0.119, 0.184, 0.133])

    def test_ubarf(self):
        assert_allclose([np.sqrt(bubble.ubarf2) for bubble in self.bubbles], self.UBARFS_REF, rtol=6.8e-3)


class ThermoTestHindmarshHijazi(ThermoTest, unittest.TestCase):
    # Input parameters
    ALPHA_NS = np.array([0.578, 0.151, 0.091])
    V_WALLS = np.array([0.5, 0.7, 0.77])
    # Reference values
    KAPPA_REF = np.array([0.610, 0.522, 0.264])
    OMEGA_REF = np.array([0.395, 0.491, 0.744])
    BVA_KE_FRAC_REF = np.array([0.223, 0.0684, 0.022])


class ThermoTestBag(ThermoTest, unittest.TestCase):
    """Test that the results have not changed due to code changes

    Reference data is has been generated with PTtools.
    """

    ALPHA_NS = np.array(np.repeat([0.1, 0.2, 0.3], 3))
    V_WALLS = np.array(np.tile([0.3, 0.7, 0.8], 3))

    KAPPA_REF = np.array([0.1227, 0.4512, 0.2346, 0.2141, 0.5645, 0.4630, 0.2881, 0.6245, 0.5786])
    OMEGA_REF = np.array([0.8773, 0.5574, 0.7683, 0.7854, 0.4408, 0.5496, 0.7113, 0.3794, 0.4305])
    BVA_KE_FRAC_REF = np.array([
        1.12012152e-02, 4.10228822e-02, 2.13351667e-02,
        3.58402474e-02, 9.40854610e-02, 7.71674313e-02,
        6.67623755e-02, 1.44117154e-01, 1.33526554e-01
    ])


class ThermoTestConstCS(ThermoTest, unittest.TestCase):
    """Test that the results have not changed due to code changes

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
