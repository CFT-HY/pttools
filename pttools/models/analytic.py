import abc

import numpy as np

from pttools.models.model import Model


class AnalyticModel(Model, abc.ABC):
    r"""A generic analytical model, where the temperature dependence of $g_\text{eff}$ is implemented directly in the
    equation of state.

    You should specify either the relativistic degrees of freedom $g_\text{eff}(\phi=s)$ and $g_\text{eff}(\phi=b)$,
    or the prefactors $a_s$ and $a_b$.
    The convention for the latter is as in :notes:`\ ` eq. 7.33. for the bag model, where
    $$p_s = a_sT^4 - V_s,$$
    $$p_b = a_bT^4 - V_b.$$

    :param V_s: $V_s = \epsilon_s$, the potential term of $p$ in the symmetric phase
    :param V_b: $V_b = \epsilon_b$, the potential term of $p$ in the broken phase
    :param a_s: prefactor of $p$ in the symmetric phase
    :param a_b: prefactor of $p$ in the broken phase
    :param g_s: $g_\text{eff}(\phi=s)$, degrees of freedom for $p$ in the symmetric phase
    :param g_b: $g_\text{eff}(\phi=b)$, degrees of freedom for $p$ in the broken phase
    :param name: custom name for the model
    """
    def __init__(
            self,
            V_s: float, V_b: float = 0,
            a_s: float = None, a_b: float = None,
            g_s: float = None, g_b: float = None,
            name: str = None):
        super().__init__(V_s=V_s, V_b=V_b, name=name)

        self.a_s: float
        self.a_b: float
        if a_s is not None and a_b is not None and g_s is None and g_b is None:
            self.a_s = a_s
            self.a_b = a_b
        elif a_s is None and a_b is None and g_s is not None and g_b is not None:
            self.a_s = np.pi**2/90 * g_s
            self.g_b = np.pi**2/90 * g_b
        else:
            raise ValueError("Specify either a_s and a_b or g_s and g_b")
