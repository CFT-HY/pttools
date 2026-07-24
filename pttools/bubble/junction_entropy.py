"""Entropy fluxes at the phase boundary"""

import numba

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
        allow_negative_entropy_flux_change: bool = False) -> tuple[bool, float, float, float]:
    """False = OK, True = fail"""
    entropy_flux1 = entropy_flux(v_tilde=v1_tilde, s=model.s(w1, phase1))
    entropy_flux2 = entropy_flux(v_tilde=v2_tilde, s=model.s(w2, phase2))
    entropy_diff = entropy_flux2 - entropy_flux1
    fail_individual = entropy_flux1 < 0 or entropy_flux2 < 0

    # phase2 - phase1 > 0 when going from symmetric to broken
    # entropy_flux2 - entropy_flux1 > 0 when going from symmetric to broken
    # -> If their product is negative, then entropy is decreasing.
    fail_total = (not allow_negative_entropy_flux_change) and (phase2 - phase1) * (entropy_flux2 - entropy_flux1) < 0

    return fail_individual or fail_total, entropy_flux1, entropy_flux2, entropy_diff


# This uses gamma(), but it's so unlikely to change, that caching this is OK.
@numba.njit(cache=True)
def entropy_flux(v_tilde: th.FloatOrArr, s: th.FloatOrArr) -> th.FloatOrArr:
    r"""Entropy flux $S$
    $$S^z = su^z = \gamma(\tilde{v}) \tilde{v} s$$
    :bhusal_2026:`\ ` eq. 41
    :notes:`\ ` p. 23,
    :maki_msc:`\ ` eq. 2.32

    :param v_tilde: $\tilde{v}$
    :param s: $s$
    """
    return gamma(v_tilde) * v_tilde * s
