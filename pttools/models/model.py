"""Template for equations of state"""

import abc

import pttools.type_hints as th


class Model:
    """Template for equations of state"""
    def __init__(self, V_s: float = 0, V_b: float = 0):
        self.V_s = V_s
        self.V_b = V_b

        # Equal values are allowed so that the default values are accepted.
        if V_b > V_s:
            raise ValueError("The bubble does not expand if V_b >= V_s.")

    @staticmethod
    @abc.abstractmethod
    def cs2(w: float, phase: float):
        """Speed of sound squared. This must be a Numba-compiled function."""
        pass

    @abc.abstractmethod
    def p(self, w: float, phase: float):
        pass

    @abc.abstractmethod
    def V(self, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Potential"""
        return phase*self.V_b + (1 - phase)*self.V_s

    @abc.abstractmethod
    def w(self, temp: th.FloatOrArr, phase: any = None):
        pass
