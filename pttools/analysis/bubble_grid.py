import typing as tp

import numpy as np

from pttools.bubble.bubble import NotYetSolvedError
from pttools.analysis.parallel import create_bubbles
if tp.TYPE_CHECKING:
    from pttools.models.model import Model


class BubbleGrid:
    def __init__(self, bubbles: np.ndarray):
        self.bubbles = bubbles

    def get_value(self, name: str, dtype: tp.Type = None) -> np.ndarray:
        with np.nditer(
                [self.bubbles, None],
                flags=("refs_ok", ),
                op_flags=[["readonly"], ["writeonly", "allocate"]],
                op_dtypes=(object, dtype)) as it:
            for bubble, res in it:
                try:
                    res[...] = getattr(bubble.item(), name)
                except NotYetSolvedError:
                    res[...] = None
            return it.operands[1]

    def elapsed(self) -> np.ndarray:
        return self.get_value("elapsed", dtype=np.float_)

    def kappa(self) -> np.ndarray:
        return self.get_value("kappa", dtype=np.float_)

    def numerical_error(self) -> np.ndarray:
        return self.get_value("numerical_error", dtype=np.bool_)

    def omega(self) -> np.ndarray:
        return self.get_value("omega", dtype=np.float_)

    def solver_failed(self) -> np.ndarray:
        return self.get_value("solver_failed", dtype=np.bool_)

    def unphysical_alpha_plus(self) -> np.ndarray:
        return self.get_value("unphysical_alpha_plus", dtype=np.bool_)

    def unphysical_entropy(self) -> np.ndarray:
        return self.get_value("unphysical_entropy", dtype=np.bool_)


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
                kwargs={"use_bag_solver": use_bag_solver}
        )
        if func is None:
            bubbles = data
        else:
            bubbles = data[0]
            self.data = data[1] if len(data) == 2 else data[1:]

        self.model = model
        self.v_walls = v_walls
        self.alpha_ns = alpha_ns

        super().__init__(bubbles)
