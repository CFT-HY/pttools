"""Unit tests for the properties of a bubble."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import DEFAULT_N_XI, Bubble
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel
from tests.utils import TEST_JSON_PATH


class BubbleTest(unittest.TestCase):
    """Unit tests for the properties of a bubble."""

    model: tp.ClassVar[BagModel]
    bubble: tp.ClassVar[Bubble]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.model = BagModel(a_s=1.1, a_b=1, V_s=1)
        cls.bubble = Bubble(cls.model, v_wall=0.5, alpha_n=0.1)

    def test_cs2_Tn_bag(self) -> None:
        r"""In the bag model $c_s^2 = 1/3$ in both phases."""
        assert self.bubble.css2_Tn == pytest.approx(1/3, abs=5e-16)
        assert self.bubble.csb2_Tn == pytest.approx(1/3, abs=5e-16)

    def test_cs2_Tn_const_cs(self) -> None:
        r"""In the constant sound speed model $c_s^2$ is constant in each phase."""
        model = ConstCSModel(css2=1/3 - 0.01, csb2=1/3 - 0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        assert bubble.css2_Tn == pytest.approx(1/3 - 0.01, abs=5e-13)
        assert bubble.csb2_Tn == pytest.approx(1/3 - 0.011, abs=5e-13)

    def test_export(self) -> None:
        """Test that the bubble can be exported to a JSON file."""
        self.bubble.export(TEST_JSON_PATH / "bubble.json")

    def test_vp_vm_tilde_ratio_giese(self) -> None:
        r"""Test that the ratio $\tilde{v}_+ / \tilde{v}_-$ of the Giese et al. method is positive."""
        assert self.bubble.vp_vm_tilde_ratio_giese > 0

    @unittest.expectedFailure
    def test_v_mu(self) -> None:
        r"""Test that the bubble has a positive $v_\mu$ (currently fails)."""
        # Todo: fix this
        assert self.bubble.v_mu > 0

    @unittest.expectedFailure
    def test_v_wall_1(self) -> None:
        """Test that a bubble cannot be created with the wall speed of light."""
        Bubble(self.model, v_wall=1, alpha_n=0.1)

    def test_v_wall_high(self) -> None:
        """Test that a bubble can be created with a wall speed close to the speed of light."""
        Bubble(self.model, v_wall=0.95, alpha_n=0.1)

    def test_v_wall_low(self) -> None:
        """Test that a bubble can be created with a very low wall speed."""
        Bubble(self.model, v_wall=0.01, alpha_n=0.1)

    def test_v_wall_low_custom_low_n_xi(self) -> None:
        """Test that the warning message for low v_wall and low n_xi is generated properly."""
        Bubble(self.model, v_wall=0.01, alpha_n=0.1, n_xi=DEFAULT_N_XI // 2)

    # -----
    # Averaged
    # -----
    def test_e_bar(self) -> None:
        """Test that the average energy density is positive."""
        assert self.bubble.e_bar > 0

    def test_kappa(self) -> None:
        """Test that the kinetic energy efficiency factor is between 0 and 1."""
        kappa = self.bubble.kappa
        assert kappa > 0
        assert kappa < 1

    def test_kappa_giese(self) -> None:
        """Test that the kinetic energy efficiency factor of the Giese et al. definition is positive."""
        assert self.bubble.kappa_giese > 0

    def test_g_star(self) -> None:
        """Test that the effective number of degrees of freedom for the energy density is positive."""
        assert self.bubble.g_star > 0

    def test_gs_star(self) -> None:
        """Test that the effective number of degrees of freedom for the entropy density is positive."""
        assert self.bubble.gs_star > 0

    def test_mean_adiabatic_index(self) -> None:
        """Test that the mean adiabatic index is positive."""
        assert self.bubble.mean_adiabatic_index > 0

    def test_nu_gdh2024(self) -> None:
        r"""Test that $\nu$ of :giombi_2024_cs:`\ ` is zero for the bag model."""
        assert self.bubble.nu_gdh2024 == pytest.approx(0, abs=5e-8)

    def test_omega(self) -> None:
        """Test that the thermal energy efficiency factor is between 0 and 1."""
        omega = self.bubble.omega
        assert omega > 0
        assert omega < 1

    def test_omega_barotropic(self) -> None:
        """Test that the barotropic thermal energy efficiency factor is between 0 and 1."""
        omega = self.bubble.omega_barotropic
        assert omega > 0
        assert omega < 1

    def test_T_star(self) -> None:
        """Test that the temperature is positive."""
        assert self.bubble.T_star > 0

    def test_ubarf(self) -> None:
        """Test that the enthalpy-weighted RMS fluid velocity and its square are positive."""
        assert self.bubble.ubarf > 0
        assert self.bubble.ubarf2 > 0

    def test_w_bar(self) -> None:
        """Test that the average enthalpy density is positive."""
        assert self.bubble.w_bar > 0

    # -----
    # bva = bubble volume averaged
    # -----
    def test_entropy_density_diff(self) -> None:
        """Test that the bubble volume averaged entropy density difference is positive."""
        assert self.bubble.entropy_density_diff > 0

    def test_entropy_density_diff_relative(self) -> None:
        """Test that the relative bubble volume averaged entropy density difference is positive."""
        assert self.bubble.entropy_density_diff_relative > 0

    def test_kinetic_energy_density(self) -> None:
        """Test that the bubble volume averaged kinetic energy density is positive."""
        assert self.bubble.kinetic_energy_density > 0

    def test_kinetic_energy_fraction(self) -> None:
        """Test that the bubble volume averaged kinetic energy fraction is between 0 and 1."""
        kef = self.bubble.kinetic_energy_fraction
        assert kef > 0
        assert kef < 1

    def test_thermal_energy_density(self) -> None:
        """Test that the bubble volume averaged thermal energy density is positive."""
        assert self.bubble.thermal_energy_density > 0

    def test_thermal_energy_fraction(self) -> None:
        """Test that the bubble volume averaged thermal energy fraction is positive."""
        tef = self.bubble.thermal_energy_fraction
        assert tef > 0
        # assert tef < 1

    def test_trace_anomaly(self) -> None:
        """Test that the bubble volume averaged trace anomaly is finite."""
        assert np.isfinite(self.bubble.trace_anomaly)

    # -----
    # va = volume averaged
    # -----
    def test_va_enthalpy_density(self) -> None:
        """Test that the volume averaged enthalpy density is positive."""
        assert self.bubble.va_enthalpy_density > 0

    def test_va_entropy_density_diff(self) -> None:
        """Test that the volume averaged entropy density difference is positive."""
        assert self.bubble.va_entropy_density_diff > 0

    def test_va_entropy_density_diff_relative(self) -> None:
        """Test that the relative volume averaged entropy density difference is positive."""
        assert self.bubble.va_entropy_density_diff_relative > 0

    def test_va_kinetic_energy_fraction(self) -> None:
        """Test that the volume averaged kinetic energy fraction is between 0 and 1."""
        kef = self.bubble.va_kinetic_energy_fraction
        assert kef > 0
        assert kef < 1

    def test_va_kinetic_energy_density(self) -> None:
        """Test that the volume averaged kinetic energy density is positive."""
        ked = self.bubble.va_kinetic_energy_density
        assert ked > 0

    def test_va_thermal_energy_density_diff(self) -> None:
        """Test that the volume averaged thermal energy density difference is positive."""
        ted = self.bubble.va_thermal_energy_density_diff
        assert ted > 0

    def test_va_thermal_energy_fraction(self) -> None:
        """Test that the volume averaged thermal energy fraction is between 0 and 1."""
        tef = self.bubble.va_thermal_energy_fraction
        assert tef > 0
        assert tef < 1

    def test_va_trace_anomaly_diff(self) -> None:
        """Test that the volume averaged trace anomaly difference is finite."""
        trace_anomaly = self.bubble.va_trace_anomaly_diff
        assert np.isfinite(trace_anomaly)
