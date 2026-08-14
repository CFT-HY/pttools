"""Holders for a grid of Bubbles of various parameter combinations."""

import typing as tp

import numpy as np
from numpy.typing import NDArray

from pttools.bubble.bubble import BubbleArr, NotYetSolvedError
from pttools.analysis.parallel import create_bubbles
import pttools.type_hints as th
if tp.TYPE_CHECKING:
    from pttools.models.model import Model


class BubbleGrid:
    """A grid of bubbles"""
    def __init__(self, bubbles: BubbleArr):
        self.bubbles = bubbles

    def get_value(self, name: str, dtype: tp.Type | None = None) -> NDArray:
        with np.nditer(
                [self.bubbles, None],
                flags=("refs_ok", ),
                op_flags=[["readonly"], ["writeonly", "allocate"]],
                op_dtypes=(object, dtype)) as it:
            for bubble_container, res in it:
                bubble = bubble_container.item()
                if bubble is None:
                    res[...] = None
                else:
                    try:
                        res[...] = getattr(bubble, name)
                    except NotYetSolvedError:
                        res[...] = None
            return it.operands[1]

    def kappa(self) -> th.FloatArr:
        return self.get_value("kappa", dtype=np.float64)

    def numerical_error(self) -> th.BoolArr:
        return self.get_value("numerical_error", dtype=np.bool_)

    def omega(self) -> th.FloatArr:
        return self.get_value("omega", dtype=np.float64)

    def solver_failed(self) -> th.BoolArr:
        return self.get_value("solver_failed", dtype=np.bool_)

    def solving_duration(self) -> th.FloatArr:
        return self.get_value("solving_duration", dtype=np.float64)

    def unphysical_alpha_plus(self) -> th.BoolArr:
        return self.get_value("unphysical_alpha_plus", dtype=np.bool_)

    def negative_net_entropy_change(self) -> th.BoolArr:
        return self.get_value("negative_net_entropy_change", dtype=np.bool_)


class BubbleGridVWAlpha(BubbleGrid):
    r"""A grid of bubbles with different $v_\text{wall}$ and $\alpha_n$ values."""
    def __init__(
            self,
            model: "Model",
            v_walls: th.FloatArr1D,
            alpha_ns: th.FloatArr1D,
            func: tp.Callable | None = None,
            use_bag_solver: bool = False):
        data = create_bubbles(
                model, v_walls, alpha_ns, func,
                kwargs={"use_bag_solver": use_bag_solver, "allow_bubble_failure": True}
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
