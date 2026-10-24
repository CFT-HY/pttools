"""Compare thermodynamic quantities of the bag model to the values in references."""

from abc import ABC
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import (
    CS2_BAG_SCALAR_PTR,
    DEFAULT_FLUID_INTEGRATE_METHOD,
    DF_DTAU_PTR_BAG,
    fluid_bag,
    thermo_bag,
)
from pttools.bubble.solution_type import SolutionType
from pttools.bubble.v_minus import v_minus
from pttools.math.assertions import assert_allclose
from pttools.type_hints import FloatArr1D
from tests.bubble.ref import RefBag, Reference, RefHindmarshHijazi, RefLectureNotes


class ThermoBagTest(Reference, ABC):
    """Compare thermodynamic results of the bag model specific functions to the values in a reference."""

    RTOL_KAPPA: tp.ClassVar[float] = 6.7e-3
    RTOL_KE_FRAC: tp.ClassVar[float] = 6.9e-3

    def test_kappa(self) -> None:
        """Test the kinetic energy efficiency factor of the bag model against the reference values."""
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i] = thermo_bag.get_kappa_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=self.RTOL_KAPPA)

    def test_kappa_de(self) -> None:
        """Test the kappa of the bag model computed with the energy change against the reference values."""
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i], _ = thermo_bag.get_kappa_de_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=self.RTOL_KAPPA)

    def test_kappa_dq(self) -> None:
        """Test the kappa of the bag model computed with the thermal energy change against the reference values."""
        kappas = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            kappas[i], _ = thermo_bag.get_kappa_dq_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(kappas, self.KAPPA_REF, rtol=self.RTOL_KAPPA)

    def test_ke_de_frac_bag(self) -> None:
        """Test the kinetic energy fraction computed together with the energy change against the reference values."""
        ke_fracs = np.zeros_like(self.KAPPA_REF)
        de_fracs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ke_fracs[i], de_fracs[i] = thermo_bag.get_ke_de_frac_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(ke_fracs, self.BVA_KE_FRAC_REF, rtol=self.RTOL_KE_FRAC)

    def test_ke_frac_bag(self) -> None:
        """Test the kinetic energy fraction of the bag model against the reference values."""
        ke_fracs = np.zeros_like(self.KAPPA_REF)
        for i in range(self.ALPHA_NS.size):
            ke_fracs[i] = thermo_bag.get_ke_frac_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])
        assert_allclose(ke_fracs, self.BVA_KE_FRAC_REF, rtol=self.RTOL_KE_FRAC)

    def test_ke_frac_new_bag(self) -> None:
        """Test the kinetic energy fraction of the new bag model function against the reference values."""
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
    r"""Compare bag model thermodynamic functions to the values in :notes:`\ `."""

    def test_ubarf2(self) -> None:
        """Test the enthalpy-weighted RMS fluid velocity of the bag model against the reference values."""
        ubarfs: FloatArr1D = np.zeros_like(self.UBARF_REF)
        for i in range(self.ALPHA_NS.size):
            ubarfs[i] = np.sqrt(thermo_bag.get_ubarf2_bag(
                v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i],
                df_dtau_ptr=DF_DTAU_PTR_BAG,
                ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_ptr=CS2_BAG_SCALAR_PTR,
                # The reference values have been computed with w_bar = w_n.
                wn_as_w_bar=True))
        assert_allclose(ubarfs, self.UBARF_REF, rtol=2.7e-3)

    def test_ubarf2_new_bag(self) -> None:
        """Test the RMS fluid velocity of the new bag model function against the reference values."""
        ubarfs: FloatArr1D = np.zeros_like(self.UBARF_REF)
        for i in range(self.ALPHA_NS.size):
            ubarfs[i] = np.sqrt(thermo_bag.get_ubarf2_new_bag(v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i]))
        assert_allclose(ubarfs, self.UBARF_REF, rtol=5.0e-2)


#: Legacy profiles of :py:func:`pttools.bubble.fluid_bag.sound_shell_bag` for each solution type:
#: ${v}_\text{wall}$, $\alpha_n$ and the solution type
DE_WALL_CASES: tuple[tuple[float, float, SolutionType], ...] = (
    (0.44, 0.1, SolutionType.SUB_DEF),
    (0.44, 0.5, SolutionType.SUB_DEF),
    (0.7, 0.1, SolutionType.HYBRID),
    (0.6, 0.05, SolutionType.HYBRID),
    (0.92, 0.1, SolutionType.DETON),
    (0.92, 0.67, SolutionType.DETON),
)


def lambda_profile(v_wall: float, alpha_n: float) -> tuple[FloatArr1D, FloatArr1D, FloatArr1D]:
    r"""$\xi$, $w$ and $\lambda = \Delta e / w_n$ of a legacy bag profile."""
    kwargs = {
        "df_dtau_ptr": DF_DTAU_PTR_BAG, "ode_method": DEFAULT_FLUID_INTEGRATE_METHOD, "cs2_ptr": CS2_BAG_SCALAR_PTR}
    _, w, xi = fluid_bag.sound_shell_bag(v_wall, alpha_n, n_xi=2000, **kwargs)
    return xi, w, thermo_bag.de_from_w_bag(w, xi, v_wall, alpha_n, **kwargs) / w[-1]


@pytest.mark.parametrize(("v_wall", "alpha_n", "sol_type"), DE_WALL_CASES)
def test_de_from_w_wall_continuous(v_wall: float, alpha_n: float, sol_type: SolutionType) -> None:
    r"""The point at $\xi = {v}_\text{wall}$ is continuous with its own side of the wall.

    In detonations the point at the wall is the end of the rarefaction wave and therefore in the broken phase,
    and in deflagrations and hybrids it's the start of the shell and therefore in the symmetric phase.
    Previously the point was always considered to be in the symmetric phase,
    which made $\lambda$ too large by $\frac{3}{4} \alpha_n$ at the wall of detonations.
    """
    xi, _, lam = lambda_profile(v_wall, alpha_n)
    i_wall = np.flatnonzero(xi == v_wall)
    assert i_wall.size == 1
    i = i_wall[0]
    i_side = i - 1 if sol_type == SolutionType.DETON else i + 1
    i_other = i + 1 if sol_type == SolutionType.DETON else i - 1
    jump_side = abs(lam[i] - lam[i_side])
    jump_other = abs(lam[i] - lam[i_other])
    assert jump_side < 0.02 * abs(lam[i])
    assert jump_side < 0.1 * jump_other


@pytest.mark.parametrize("alpha_n", [0.1, 0.67])
def test_de_from_w_detonation_junction(alpha_n: float) -> None:
    r"""$\lambda$ behind the wall of a detonation agrees with the junction conditions of the bag model.

    The fluid ahead of a detonation is at rest at $w_{n}$,
    so $\tilde{v}_{+} = {v}_\text{wall}$ and $\alpha_{+} = \alpha_n$.
    $$\frac{w_{-}}{w_{n}} = \frac{\gamma_{+}^2 \tilde{v}_{+}}{\gamma_{-}^2 \tilde{v}_{-}}, \quad
    \lambda_{-} = \frac{3}{4} \frac{w_{-}}{w_{n}} - \frac{3}{4} (1 + \alpha_n)$$
    """
    v_wall = 0.92
    xi, w, lam = lambda_profile(v_wall, alpha_n)
    i_wall = np.flatnonzero(xi == v_wall)[0]
    vm = v_minus(vp=v_wall, ap=alpha_n, sol_type=SolutionType.DETON)
    w_ratio = (v_wall / (1 - v_wall**2)) / (vm / (1 - vm**2))
    assert w[i_wall] / w[-1] == pytest.approx(w_ratio, rel=1e-6)
    assert lam[i_wall] == pytest.approx(0.75 * w_ratio - 0.75 * (1 + alpha_n), rel=1e-6)
