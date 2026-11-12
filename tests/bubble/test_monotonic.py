"""Reference tests for the solutions whose profiles approach the fixed point at the sound speed."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.bubble.solution_type import SolutionType
from pttools.models import BagModel, ConstCSModel
from pttools.models.model import Model
from pttools.ssm.nucleation import nucleation_f

#: Parameters of the constant sound speed models
MODEL_KWARGS: dict[str, tp.Any] = {"a_s": 1.5, "a_b": 1, "V_s": 1, "alpha_n_min": 0.05}

#: Reference values of the quantities of the bubbles, computed with the generic fluid solver
#: before the profiles were cut where they reach the fixed point at the sound speed:
#: the rarefaction wave behind the wall of the hybrids,
#: and the profile in front of the wall of a deflagration with a very weak shock.
#: The cut removed only the numerical fluctuations around the fixed point,
#: and changed these values by less than 2e-9 relatively, except $v_\text{sh}$ by 6e-9.
#: ``nucleation_f`` is :py:func:`pttools.ssm.nucleation.nucleation_f` for $\tilde{\beta} = 100$.
REFERENCE: dict[str, dict[str, float]] = {
    "bag": {
        "kappa": 0.4116778067115894, "omega": 0.5920787501206762, "ubarf2": 0.029486980811130048,
        "w_bar": 13.961341425507996, "kinetic_energy_fraction": 0.03742525515559904,
        "va_enthalpy_density": 13.960923960553169, "mean_adiabatic_index": 1.2692128568643637,
        "T_star": 1.3668269792834509, "nucleation_f": 0.2504866358579222,
    },
    "const_cs": {
        "kappa": 0.4521341430140255, "omega": 0.5540724862869134, "ubarf2": 0.028750363162501284,
        "w_bar": 16.368840539491096, "kinetic_energy_fraction": 0.03461813986809907,
        "va_enthalpy_density": 16.367587304868223, "mean_adiabatic_index": 1.2040940030020577,
        "T_star": 1.2841752915395288, "nucleation_f": 0.217774629765918,
    },
    "const_cs_strong": {
        "kappa": 0.7209050414934559, "omega": 0.28011722974183445, "ubarf2": 0.283043121467348,
        "w_bar": 2.5088678609553567, "kinetic_energy_fraction": 0.25209726271583055,
        "va_enthalpy_density": 2.5093764794080524, "mean_adiabatic_index": 0.8906673351004315,
        "T_star": 0.8728705517056153, "nucleation_f": 0.5387824923508645,
    },
    "const_cs_2": {
        "kappa": 0.4703990633190987, "omega": 0.5322331951233381, "ubarf2": 0.06342409475469435,
        "w_bar": 5.839498390615365, "kinetic_energy_fraction": 0.0731188906785029,
        "va_enthalpy_density": 5.839187929215903, "mean_adiabatic_index": 1.1528566700290348,
        "T_star": 1.0315172373084238, "nucleation_f": 0.32792804994295993,
    },
    "weak_shock_deflagration": {
        "kappa": 0.10891846824131847, "omega": 0.8937402320190723, "ubarf2": 0.008086374628760546,
        "w_bar": 13.462037657963712, "kinetic_energy_fraction": 0.00989627997897971,
        "va_enthalpy_density": 13.461654788077116, "mean_adiabatic_index": 1.2238216052694288,
        "T_star": 1.354439511592807, "v_sh": 0.577350270126357, "nucleation_f": 4.3502947868299895,
    },
}

#: The bubbles of :py:data:`REFERENCE` that are hybrids
HYBRIDS: tuple[str, ...] = ("bag", "const_cs", "const_cs_strong", "const_cs_2")


def create_bubbles() -> dict[str, Bubble]:
    """Create the bubbles of :py:data:`REFERENCE`."""
    cases: dict[str, tuple[Model, float, float]] = {
        "bag": (BagModel.cached(alpha_n_min=0.01), 0.6, 0.1),
        "const_cs": (ConstCSModel.cached(css2=1/4, csb2=7/27, **MODEL_KWARGS), 0.55, 0.1),
        "const_cs_strong": (ConstCSModel.cached(css2=1/4, csb2=7/27, **MODEL_KWARGS), 0.55, 0.5994842503189409),
        "const_cs_2": (ConstCSModel.cached(css2=1/3, csb2=1/4, **MODEL_KWARGS), 0.6, 0.2),
    }
    bubbles = {
        name: Bubble.cached(model, v_wall=v_wall, alpha_n=alpha_n)
        for name, (model, v_wall, alpha_n) in cases.items()
    }
    # The fluctuations near the fixed point appeared with the n_xi of the datasets.
    bubbles["weak_shock_deflagration"] = Bubble.cached(
        ConstCSModel.cached(css2=1/3, csb2=1/3, **MODEL_KWARGS), v_wall=0.275, alpha_n=0.1, n_xi=10000)
    return bubbles


class MonotonicTest(unittest.TestCase):
    """Reference tests for the solutions whose profiles approach the fixed point at the sound speed."""

    bubbles: tp.ClassVar[dict[str, Bubble]]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        """Solve the bubbles."""
        cls.bubbles = create_bubbles()

    def test_monotonic(self) -> None:
        r"""$\xi$ should be monotonic, also near the fixed point at the sound speed."""
        for name, bubble in self.bubbles.items():
            with self.subTest(name=name):
                assert bubble.sol_type == (SolutionType.HYBRID if name in HYBRIDS else SolutionType.SUB_DEF)
                assert np.all(np.diff(bubble.xi) >= 0)
                assert not bubble.non_monotonic_xi
                assert not bubble.failed or name in ("const_cs_strong", "const_cs_2")

    def test_reference(self) -> None:
        """The quantities of the bubbles should match the reference values."""
        for name, bubble in self.bubbles.items():
            for quantity, ref in REFERENCE[name].items():
                with self.subTest(name=name, quantity=quantity):
                    value = nucleation_f(bubble.xi, bubble.T, 100., bubble.v_wall) if quantity == "nucleation_f" \
                        else getattr(bubble, quantity)
                    assert value == pytest.approx(ref, rel=1e-8)

    def test_wall(self) -> None:
        r"""The profile of a hybrid should have a point at $\xi = v_{\text{wall}}$ on both sides of the wall."""
        for name in HYBRIDS:
            bubble = self.bubbles[name]
            with self.subTest(name=name):
                assert np.count_nonzero(bubble.xi == bubble.v_wall) == 2


if __name__ == "__main__":
    unittest.main()
