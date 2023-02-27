import typing as tp

import numpy as np

from pttools.analysis.parallel import create_bubbles
if tp.TYPE_CHECKING:
    from pttools.models.model import Model


class BubbleGrid:
    def __init__(self, bubbles: np.ndarray):
        self.bubbles = bubbles

    def get_value(self, name: str) -> np.ndarray:
        with np.nditer([self.bubbles, None], flags=("refs_ok", )) as it:
            for bubble, res in it:
                res[...] = getattr(bubble.item(), name)
            return it.operands[1]

    def numerical_error(self):
        return self.get_value("numerical_error")

    def unphysical_alpha_plus(self):
        return self.get_value("unphysical_alpha_plus")

    def unphysical_entropy(self):
        return self.get_value("unphysical_entropy")


class BubbleGridVWAlpha(BubbleGrid):
    def __init__(self, model: "Model", v_walls: np.ndarray, alpha_ns: np.ndarray, func: callable = None):
        data = create_bubbles(model, v_walls, alpha_ns, func, output_dtypes=(object, np.float_))
        if func is None:
            bubbles = data
        else:
            bubbles = data[0]
            self.data = data[1]
        super().__init__(bubbles)
