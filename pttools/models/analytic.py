import abc

from pttools.models.base import Model


class AnalyticModel(Model, abc.ABC):
    def __init__(self, a_s: float, a_b: float, V_s: float, V_b: float = 0):
        r"""
        :param a_s: prefactor of $p$ in the symmetric phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param a_b: prefactor of $p$ in the broken phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param V_s: $V_s \equiv \epsilon_s$, the potential term of $p$ in the symmetric phase
        :param V_b: $V_b \equiv \epsilon_b$, the potential term of $p$ in the broken phase
        """
        super().__init__(V_s=V_s, V_b=V_b)
        self.a_s = a_s
        self.a_b = a_b
