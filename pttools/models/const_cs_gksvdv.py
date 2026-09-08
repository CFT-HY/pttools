r"""Constant $c_s$ models used in :giese_2021:`\ `, fig. 2."""

from functools import cache

import numpy as np

from pttools.models.const_cs import ConstCSModel
from pttools.type_hints import FloatArr1D

GKSVDV_ALPHA_N: FloatArr1D = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])


@cache
def gksvdv_models(
        a_s: float = 5,
        a_b: float = 1,
        V_s: float = 1,
        alpha_n_min: float = GKSVDV_ALPHA_N[0]) -> list[ConstCSModel]:
    return [
        ConstCSModel(css2=1 / 3, csb2=1 / 3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min),
        ConstCSModel(css2=1 / 3, csb2=1 / 4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min),
        ConstCSModel(css2=1 / 4, csb2=1 / 3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min),
        ConstCSModel(css2=1 / 4, csb2=1 / 4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n_min),
    ]


def gksvdv_v_wall(n: int) -> FloatArr1D:
    return np.linspace(0.2, 0.95, n)
