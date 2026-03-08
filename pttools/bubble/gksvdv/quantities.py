import numba
import numpy as np

from pttools.bubble.gksvdv.gksvdv21 import kappaNuMuModel
import pttools.type_hints as th
from pttools.speedup import NUMBA_ENABLE_CACHE


# @numba.njit(nogil=True, cache=NUMBA_ENABLE_CACHE)
def kappa_gksvdv(params: th.FloatArr1D, css2: float, csb2: float) -> float:
    r"""Compute $\kappa$ with the :giese_2021:`\ ` solver"""
    v_wall, alpha_tbn_giese = params
    try:
        kappa, v_arr, wow_arr, xi_arr, mode, vp, vm = kappaNuMuModel(
            # cs2s=model.cs2(model.w_crit, Phase.SYMMETRIC),
            # cs2b=model.cs2(model.w_crit, Phase.BROKEN),
            cs2s=css2,
            cs2b=csb2,
            al=alpha_tbn_giese,
            vw=v_wall
        )
    except ValueError:
        return np.nan
    return kappa
