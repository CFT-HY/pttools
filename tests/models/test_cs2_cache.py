"""Tests for the caching of the compiled $c_s^2$ functions of the models."""

import pickle
import unittest

import numpy as np

from pttools import models
from pttools.bubble.integrate import differentials
from pttools.bubble.phase import Phase
from pttools.speedup import DifferentialPointer


def df_dtau_cs2(df_dtau_ptr: DifferentialPointer, phase: Phase) -> float:
    r"""Extract the $c_s^2$ that the differential equation of the fluid profile uses.

    $\frac{dv}{d\tau} = 2 v c_s^2 (1 - v^2) (1 - \xi v)$,
    which can be solved for $c_s^2$ at an arbitrary point.
    """
    v, w, xi = 0.5, 1., 0.7
    du = differentials.get_solve_ivp(df_dtau_ptr)(0., np.array([v, w, xi]), np.array([phase.value, 0.]))
    return du[0] / (2 * v * (1 - v**2) * (1 - xi * v))


class TestConstCSFuncs(unittest.TestCase):
    """Tests for sharing the compiled functions between ConstCSModels with the same sound speeds."""

    CSS2 = 1/3
    CSB2 = 0.25

    @staticmethod
    def create_model(csb2: float) -> models.ConstCSModel:
        return models.ConstCSModel(css2=TestConstCSFuncs.CSS2, csb2=csb2, a_s=2, a_b=1, V_s=0.1, log_info=False)

    def test_shared(self):
        """Models with the same sound speeds share the compiled functions and the pointers."""
        model1 = self.create_model(self.CSB2)
        model2 = self.create_model(self.CSB2)
        self.assertIsNot(model1, model2)
        self.assertIs(model1.cs2, model2.cs2)
        self.assertIs(model1.cs2_neg, model2.cs2_neg)
        self.assertEqual(model1.cs2_ptr(), model2.cs2_ptr())
        self.assertEqual(model1.df_dtau_ptr(), model2.df_dtau_ptr())

    def test_not_shared(self):
        """Models with different sound speeds have their own compiled functions."""
        model1 = self.create_model(self.CSB2)
        model2 = self.create_model(0.3)
        self.assertIsNot(model1.cs2, model2.cs2)
        self.assertNotEqual(model1.cs2_ptr(), model2.cs2_ptr())
        self.assertNotEqual(model1.df_dtau_ptr(), model2.df_dtau_ptr())

    def test_pickle(self):
        """Unpickling a model restores the shared functions instead of creating new ones."""
        model1 = self.create_model(self.CSB2)
        model2 = pickle.loads(pickle.dumps(model1))
        self.assertIs(model1.cs2, model2.cs2)
        self.assertIs(model1.cs2_neg, model2.cs2_neg)
        self.assertEqual(model1.cs2_ptr(), model2.cs2_ptr())
        self.assertEqual(model1.df_dtau_ptr(), model2.df_dtau_ptr())
        self.assertEqual(model1.id, model2.id)


class TestDfDtauIdentity(unittest.TestCase):
    """Tests for the differential equations of the fluid profiles corresponding to the right models.

    The differentials used to be identified by Python's id() of the model,
    which is reused after the model has been garbage collected.
    Therefore, a model created after a previous one had been discarded
    could end up using the differential equation of the previous model.
    """

    CSB2_VALUES = (0.25, 0.26, 0.27, 0.28, 0.29, 0.3)

    def test_const_cs(self):
        for csb2 in self.CSB2_VALUES:
            model = models.ConstCSModel(css2=1/3, csb2=csb2, a_s=2, a_b=1, V_s=0.1, log_info=False)
            self.assertAlmostEqual(df_dtau_cs2(model.df_dtau_ptr(), Phase.BROKEN), csb2)
            self.assertAlmostEqual(df_dtau_cs2(model.df_dtau_ptr(), Phase.SYMMETRIC), 1/3)

    def test_full(self):
        """The general models are identified by a unique id instead of the sound speeds."""
        for css2 in (0.4**2, 0.3):
            thermo = models.ConstCSThermoModel(css2=css2, csb2=1/3, a_s=1.2, a_b=1.1, V_s=1.3)
            model = models.FullModel(thermo=thermo)
            # The cs2 of the FullModel is based on a spline, and therefore the precision is limited.
            self.assertAlmostEqual(df_dtau_cs2(model.df_dtau_ptr(), Phase.SYMMETRIC), css2, places=3)


if __name__ == "__main__":
    unittest.main()
