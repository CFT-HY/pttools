"""Entropy fluxes at the phase boundary"""

from pttools.bubble.relativity import gamma
from pttools.bubble.phase import Phase
import pttools.type_hints as th


def check_entropy_fluxes(
        model: "Model",
        v1_tilde: float,
        v2_tilde: float,
        w1: float,
        w2: float,
        phase1: Phase,
        phase2: float,
        allow_negative_entropy_flux_change: bool = False) -> tuple[bool, float, float]:
    """False = OK, True = fail"""
    s1 = model.s(w1, phase1)
    s2 = model.s(w2, phase2)
    entropy_flux1 = entropy_flux(v1_tilde, s1)
    entropy_flux2 = entropy_flux(v2_tilde, s2)
    fail_individual = entropy_flux1 < 0 or entropy_flux2 < 0
    if allow_negative_entropy_flux_change:
        fail_total = False
    else:
        fail_total = (
            (phase1 == Phase.SYMMETRIC and phase2 == Phase.BROKEN and entropy_flux1 - entropy_flux2 < 0) or
            (phase1 == Phase.BROKEN and phase2 == Phase.SYMMETRIC and entropy_flux2 - entropy_flux1 < 0)
        )
    # fail_total = False
    return fail_individual or fail_total, entropy_flux1, entropy_flux2


def entropy_flux(v_tilde: th.FloatOrArr, s: th.FloatOrArr) -> th.FloatOrArr:
    r"""Entropy flux $\gamma(\tilde{v}) \tilde{v} s$"""
    return gamma(v_tilde) * v_tilde * s
