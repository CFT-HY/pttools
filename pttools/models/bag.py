"""Bag model"""

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from .model import Model


class BagModel(Model):
    r"""Bag equation of state.
    Each integration corresponds to a line on the figure below (fig. 9 of :gw_pt_ssm:`\ `).

    .. plot:: fig/xi_v_plane.py

    """
    def __init__(self, a_s, a_b, V_s):
        super().__init__()
        self.a_s = a_s
        self.a_b = a_b
        self.V_s = V_s

    def p(self, temp, phase):
        """:notes: (eq. 7.33)"""
        if phase == Phase.SYMMETRIC:
            return self.a_s * temp**4 - self.V_s
        elif phase == Phase.BROKEN:
            return self.a_b * temp**4
        raise ValueError(f"Unknown phase: {phase}")

    def cs2(self, w: th.FloatOrArr = None, **kwargs):
        return 1/3
