import typing as tp

import numpy as np

from pttools.bubble.bubble import NotYetSolvedError
from pttools.analysis.parallel import create_bubbles
if tp.TYPE_CHECKING:
    from pttools.models.model import Model


class BubbleGrid:
    def __init__(self, bubbles: np.ndarray):
        self.bubbles = bubbles

    def get_value(self, name: str, is_obj: bool = False) -> np.ndarray:
        if is_obj:
            flags = ("refs_ok", )
        else:
            flags = ()
        with np.nditer([self.bubbles, None], flags=flags) as it:
            for bubble, res in it:
                try:
                    res[...] = getattr(bubble.item(), name)
                except NotYetSolvedError:
                    res[...] = None
            return it.operands[1]

    def kappa(self):
        return self.get_value("kappa", is_obj=True)

    def numerical_error(self):
        return self.get_value("numerical_error", is_obj=True)

    def omega(self):
        return self.get_value("omega", is_obj=True)

    def solver_failed(self):
        return self.get_value("solver_failed", is_obj=True)

    def unphysical_alpha_plus(self):
        return self.get_value("unphysical_alpha_plus", is_obj=True)

    def unphysical_entropy(self):
        return self.get_value("unphysical_entropy", is_obj=True)


class BubbleGridVWAlpha(BubbleGrid):
    def __init__(
            self,
            model: "Model",
            v_walls: np.ndarray,
            alpha_ns: np.ndarray,
            func: callable = None,
            use_bag_solver: bool = False):
        data = create_bubbles(
                model, v_walls, alpha_ns, func,
                output_dtypes=(object, np.float_),
                kwargs={"use_bag_solver": use_bag_solver}
        )
        self.v_walls = v_walls
        self.alpha_ns = alpha_ns
        if func is None:
            bubbles = data
        else:
            bubbles = data[0]
            self.data = data[1]
        super().__init__(bubbles)
