r"""$\mu, \nu$-model"""

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from .model import Model


class MuNuModel(Model):
    r"""$\mu, \nu$-model


    """
    def __init__(self, a_s: float, a_b: float, css2: float, csb2: float, eps: float):
        super().__init__()
        self.a_s = a_s
        self.a_b = a_b
        self.css2 = css2
        self.csb2 = csb2
        self.eps = eps

        self.mu = 1 + 1/css2
        self.nu = 1 + 1/csb2

    def cs2(self, phase: Phase) -> float:
        if phase == Phase.SYMMETRIC:
            return self.css2
        if phase == Phase.BROKEN:
            return self.csb2
        raise ValueError(f"Unknown phase: {phase}")

    def p(self, T: th.FloatOrArr, phase: Phase) -> th.FloatOrArr:
        if phase == Phase.SYMMETRIC:
            return 1/3 * self.a_s * T**self.mu - self.eps
        if phase == Phase.BROKEN:
            return 1/3 * self.a_b * T**self.nu
        raise ValueError(f"Unknown phase: {phase}")

    def e(self, T: th.FloatOrArr, phase: Phase) -> th.FloatOrArr:
        if phase == Phase.SYMMETRIC:
            # There is a typo in :giese_2021:. The 4 there should be a mu.
            return 1/3 * self.a_s * (self.mu - 1) * T**self.mu + self.eps
        if phase == Phase.BROKEN:
            return 1/3 * self.a_b * (self.nu - 1) * T**self.nu
        raise ValueError(f"Unknown phase: {phase}")
