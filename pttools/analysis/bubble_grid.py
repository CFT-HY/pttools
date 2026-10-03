"""Holders for a grid of Bubbles of various parameter combinations."""

import typing as tp

import numpy as np
from numpy.typing import NDArray

from pttools.analysis.parallel import create_bubbles
from pttools.bubble.bubble import BubbleArr, NotYetSolvedError
import pttools.type_hints as th

if tp.TYPE_CHECKING:
    from pttools.models.model import Model


class BubbleGrid:
    """A grid of bubbles."""

    def __init__(self, bubbles: BubbleArr):
        """:param bubbles: array of bubbles, where the unsolvable parameter combinations can be ``None``"""
        self.bubbles: BubbleArr = bubbles

    def get_value(self, name: str, dtype: type | None = None) -> NDArray:
        """Get the value of an attribute of each bubble in the grid.

        The value is ``None`` for the bubbles that are ``None`` or have not been solved.

        :param name: name of the bubble attribute
        :param dtype: data type of the output array
        :return: array of the attribute values with the same shape as the grid
        """
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
        r"""$\kappa$, kinetic efficiency factor of each bubble."""
        return self.get_value("kappa", dtype=np.float64)

    def numerical_error(self) -> th.BoolArr:
        r"""Whether each bubble has a numerical error, e.g. $\kappa + \omega \neq 1$."""
        return self.get_value("numerical_error", dtype=np.bool_)

    def omega(self) -> th.FloatArr:
        r"""$\omega$, thermal efficiency factor of each bubble."""
        return self.get_value("omega", dtype=np.float64)

    def solver_failed(self) -> th.BoolArr:
        """Whether the solver failed for each bubble."""
        return self.get_value("solver_failed", dtype=np.bool_)

    def solving_duration(self) -> th.FloatArr:
        """Time taken by the solver for each bubble in seconds."""
        return self.get_value("solving_duration", dtype=np.float64)

    def unphysical_alpha_plus(self) -> th.BoolArr:
        r"""Whether each bubble has an unphysical $\alpha_+$."""
        return self.get_value("unphysical_alpha_plus", dtype=np.bool_)

    def negative_net_entropy_change(self) -> th.BoolArr:
        """Whether the net entropy change of each bubble is negative."""
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
        r"""Create and solve the bubbles for all combinations of $v_\text{wall}$ and $\alpha_n$.

        :param model: equation of state model
        :param v_walls: $v_\text{wall}$, wall speeds
        :param alpha_ns: $\alpha_n$, transition strengths at the nucleation temperature
        :param func: function to be applied to each bubble after solving it.
            Its outputs are stored in ``self.data``.
        :param use_bag_solver: whether to use the bag model solver
        """
        data = create_bubbles(
                model, v_walls, alpha_ns, func,
                kwargs={"use_bag_solver": use_bag_solver, "allow_bubble_failure": True}
        )
        if func is None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Expected an array of bubbles, got: {type(data)}")
            bubbles = data
        else:
            bubbles = data[0]
            func_outputs = data[1:]
            self.data: NDArray | tuple[NDArray, ...] = func_outputs[0] if len(func_outputs) == 1 else func_outputs

        self.model: Model = model
        self.v_walls: th.FloatArr1D = v_walls
        self.alpha_ns: th.FloatArr1D = alpha_ns

        super().__init__(bubbles)
