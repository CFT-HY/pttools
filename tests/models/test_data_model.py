"""Tests for the DataModel, which is based on tabulated data."""

import typing as tp
import unittest

import numpy as np

from pttools import models
from pttools.bubble.phase import Phase
from pttools.type_hints import FloatArr1D


class DataModelTest(unittest.TestCase):
    """Tests for the DataModel using tabulated data of a constant sound speed model.

    The sound speeds of the phases are different, so that mixing up the phases is detected.
    The splines are linear, so the data points should be reproduced exactly.
    """

    CSS2: tp.ClassVar[float] = 1/3 - 0.01
    CSB2: tp.ClassVar[float] = 1/3 - 0.05

    ref: tp.ClassVar[models.ConstCSModel]
    model: tp.ClassVar[models.DataModel]
    temp: tp.ClassVar[FloatArr1D]
    data: tp.ClassVar[dict[Phase, dict[str, FloatArr1D]]]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.ref = models.ConstCSModel(css2=cls.CSS2, csb2=cls.CSB2, a_s=1.5, a_b=1, V_s=1, V_b=0)
        cls.temp = np.logspace(-1, 1, 41)
        cls.data = {
            phase: {
                "p": cls.ref.p_temp(cls.temp, phase),
                "e": cls.ref.e_temp(cls.temp, phase),
                "cs2": np.full_like(cls.temp, cls.CSB2 if phase == Phase.BROKEN else cls.CSS2),
            }
            for phase in (Phase.SYMMETRIC, Phase.BROKEN)
        }
        sym = cls.data[Phase.SYMMETRIC]
        brk = cls.data[Phase.BROKEN]
        cls.model = models.DataModel(
            T_s=cls.temp, T_b=cls.temp,
            p_s=sym["p"], p_b=brk["p"],
            e_s=sym["e"], e_b=brk["e"],
            cs2_s=sym["cs2"], cs2_b=brk["cs2"],
            T_crit=cls.ref.T_crit,
            name="data_const_cs"
        )

    def test_p_e_w_temp(self) -> None:
        """Test that the pressure, energy density and enthalpy correspond to the data of each phase."""
        for phase, data in self.data.items():
            with self.subTest(phase=phase):
                np.testing.assert_allclose(self.model.p_temp(self.temp, phase), data["p"], rtol=1e-12)
                np.testing.assert_allclose(self.model.e_temp(self.temp, phase), data["e"], rtol=1e-12)
                np.testing.assert_allclose(self.model.w(self.temp, phase), data["p"] + data["e"], rtol=1e-12)

    def test_temp(self) -> None:
        """Test that the temperature as a function of the enthalpy corresponds to the data of each phase."""
        for phase, data in self.data.items():
            with self.subTest(phase=phase):
                np.testing.assert_allclose(self.model.temp(data["p"] + data["e"], phase), self.temp, rtol=1e-12)

    def test_mixed_phase(self) -> None:
        """Test that an array of phases selects the data of the corresponding phase for each element."""
        phase = np.array([(1 + (-1)**i) / 2 for i in range(self.temp.size)])
        brk = phase == Phase.BROKEN
        p = np.where(brk, self.data[Phase.BROKEN]["p"], self.data[Phase.SYMMETRIC]["p"])
        e = np.where(brk, self.data[Phase.BROKEN]["e"], self.data[Phase.SYMMETRIC]["e"])
        np.testing.assert_allclose(self.model.p_temp(self.temp, phase), p, rtol=1e-12)
        np.testing.assert_allclose(self.model.e_temp(self.temp, phase), e, rtol=1e-12)
        np.testing.assert_allclose(self.model.temp(p + e, phase), self.temp, rtol=1e-12)

    def test_reference_model(self) -> None:
        r"""Test that the enthalpy corresponds to the reference model between the data points.

        The tolerance is set by the accuracy of the linear interpolation in $\log T$.
        The enthalpies of the phases differ by more than that at most temperatures,
        so mixing up the phases is detected.
        """
        rtol = 5e-2
        temp = np.sqrt(self.temp[1:] * self.temp[:-1])
        for phase in (Phase.SYMMETRIC, Phase.BROKEN):
            with self.subTest(phase=phase):
                np.testing.assert_allclose(self.model.w(temp, phase), self.ref.w(temp, phase), rtol=rtol)
        ratio = self.ref.w(temp, Phase.SYMMETRIC) / self.ref.w(temp, Phase.BROKEN)
        assert np.mean(np.abs(ratio - 1) > rtol) > 0.8

    def test_cs2(self) -> None:
        r"""Test that $c_s^2$ corresponds to the data of each phase, also for an array of phases."""
        w = np.geomspace(self.model.w_min, self.model.w_max, 20)[1:-1]
        for phase, cs2 in ((Phase.SYMMETRIC, self.CSS2), (Phase.BROKEN, self.CSB2)):
            with self.subTest(phase=phase):
                np.testing.assert_allclose(self.model.cs2(w, phase), cs2, rtol=1e-12)
        phase_arr = np.array([(1 + (-1)**i) / 2 for i in range(w.size)])
        np.testing.assert_allclose(
            self.model.cs2(w, phase_arr), np.where(phase_arr == Phase.BROKEN, self.CSB2, self.CSS2), rtol=1e-12)

    def test_cs2_out_of_range(self) -> None:
        r"""Test that $c_s^2$ is NaN outside the range of the data for both scalars and arrays."""
        w_min = self.model.w_min
        w_max = self.model.w_max
        w = np.array([w_min / 2, np.sqrt(w_min * w_max), 2 * w_max])
        for phase in (Phase.SYMMETRIC, Phase.BROKEN):
            with self.subTest(phase=phase):
                cs2 = self.model.cs2(w, phase)
                assert np.isnan(cs2[0])
                assert not np.isnan(cs2[1])
                assert np.isnan(cs2[2])
                for i, w_i in enumerate(w):
                    np.testing.assert_equal(self.model.cs2(float(w_i), phase), cs2[i])

    def test_temperature_is_physical(self) -> None:
        """Test that the temperature properties of the model are those given to the constructor."""
        assert not self.model.temperature_is_physical
        assert self.model.temperature_unit_gev == 1
        model = models.DataModel(
            T_s=self.temp, T_b=self.temp,
            p_s=self.data[Phase.SYMMETRIC]["p"], p_b=self.data[Phase.BROKEN]["p"],
            e_s=self.data[Phase.SYMMETRIC]["e"], e_b=self.data[Phase.BROKEN]["e"],
            cs2_s=self.data[Phase.SYMMETRIC]["cs2"], cs2_b=self.data[Phase.BROKEN]["cs2"],
            T_crit=self.ref.T_crit, T_is_physical=True, T_unit_gev=1e-3,
            name="data_const_cs_physical"
        )
        assert model.temperature_is_physical
        assert model.temperature_unit_gev == 1e-3
