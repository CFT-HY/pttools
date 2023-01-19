import typing as tp
import unittest

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.models.bag import BagModel
from tests.utils.test_assertions import assert_allclose


class ThermoTest:
    ALPHA_NS: np.ndarray
    V_WALLS: np.ndarray

    KAPPA_REF: np.ndarray
    OMEGA_REF: np.ndarray
    KE_FRAC_REF: np.ndarray

    bubbles: tp.List[Bubble]

    @classmethod
    # pylint: disable=invalid-name
    def setUpClass(cls) -> None:
        model = BagModel(a_s=1.1, a_b=1, V_s=1)
        cls.bubbles = [
            Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
            for v_wall, alpha_n in zip(cls.V_WALLS, cls.ALPHA_NS)
        ]
        for bubble in cls.bubbles:
            bubble.solve()

    def test_kappa(self):
        assert_allclose([bubble.kappa for bubble in self.bubbles], self.KAPPA_REF, rtol=1.5e-2)

    def test_ke_frac(self):
        assert_allclose([bubble.kinetic_energy_fraction for bubble in self.bubbles], self.KE_FRAC_REF, rtol=1.5e-2)

    def test_omega(self):
        assert_allclose([bubble.omega for bubble in self.bubbles], self.OMEGA_REF, rtol=1.3e-2)


class ThermoTestLectureNotes(ThermoTest, unittest.TestCase):
    # Input parameters
    ALPHA_NS = np.array([0.1, 0.1, 0.1])
    V_WALLS = np.array([0.4, 0.7, 0.8])
    # Reference values
    KAPPA_REF = np.array([0.189, 0.452, 0.235])
    OMEGA_REF = np.array([0.815, 0.559, 0.769])
    KE_FRAC_REF = np.array([0.0172, 0.0411, 0.0213])
    UBARFS_REF = np.array([0.119, 0.184, 0.133])

    def test_ubarf(self):
        assert_allclose([np.sqrt(bubble.ubarf2) for bubble in self.bubbles], self.UBARFS_REF, rtol=6.8e-3)


class ThermoBagTestHindmarshHijazi(ThermoTest, unittest.TestCase):
    # Input parameters
    ALPHA_NS = np.array([0.578, 0.151, 0.091])
    V_WALLS = np.array([0.5, 0.7, 0.77])
    # Reference values
    KAPPA_REF = np.array([0.610, 0.522, 0.264])
    OMEGA_REF = np.array([0.395, 0.491, 0.744])
    KE_FRAC_REF = np.array([0.223, 0.0684, 0.022])
