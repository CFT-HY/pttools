"""Utilities for handling functions for the differential equations"""

import threading
import typing as tp

import numba
import numpy as np

from pttools.speedup.numba_wrapper import CFunc, CPUDispatcher, lsoda_sig
from pttools.speedup.options import NUMBA_DISABLE_JIT

Differential = tp.Union[tp.Callable[[float, np.ndarray, np.ndarray, tp.Optional[np.ndarray]], None], CFunc]
DifferentialOdeint = tp.Union[tp.Callable[[np.ndarray, float, tp.Optional[np.ndarray]], np.ndarray], CPUDispatcher]
DifferentialSolveIVP = tp.Union[tp.Callable[[float, np.ndarray, tp.Optional[np.ndarray]], np.ndarray], CPUDispatcher]
DifferentialPointer = numba.types.CPointer(lsoda_sig)


class DifferentialCache:
    """Cache for the functions that compute the differentials

    This cache system automatically compiles versions for
    :func:`scipy.integrate.odeint`,
    :func:`scipy.integrate.solve_ivp`
    and NumbaLSODA.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cache_cfunc: tp.Dict[DifferentialPointer, Differential] = {}
        self._cache_odeint: tp.Dict[DifferentialPointer, DifferentialOdeint] = {}
        self._cache_solve_ivp: tp.Dict[DifferentialPointer, DifferentialSolveIVP] = {}

    def add(
            self,
            name: str, differential: Differential,
            p_last_is_backwards: bool = True,
            ndim: int = 3) -> DifferentialPointer:
        with self._lock:
            if name in self._cache_cfunc:
                raise ValueError("The key is already in the cache")
            differential_njit = numba.njit(differential)
            # differential_core = differential_cfunc if NUMBA_DISABLE_JIT else differential_njit
            if not NUMBA_DISABLE_JIT:
                differential_cfunc = numba.cfunc(lsoda_sig)(differential)
                if p_last_is_backwards:
                    @numba.cfunc(lsoda_sig)
                    def differential_numbalsoda(t: float, u: np.ndarray, du: np.ndarray, p: np.ndarray):
                        differential_njit(t, u, du, p)
                        # TODO: implement support for arbitrarily long p
                        # This cannot be used when jitting is disabled
                        # https://github.com/numba/numba/issues/8002
                        p_arr = numba.carray(p, (3,), numba.types.double)
                        if p_arr[-1]:
                            for i in range(ndim):
                                du[i] *= -1.
                else:
                    differential_numbalsoda = differential_cfunc

            @numba.njit
            def differential_odeint(y: np.ndarray, t: float, p: np.ndarray = None) -> np.ndarray:
                du = np.empty_like(y)
                differential_njit(t, y, du, p)
                return du

            @numba.njit
            def differential_solve_ivp(t: float, y: np.ndarray, p: np.ndarray = None) -> np.ndarray:
                du = np.empty_like(y)
                differential_njit(t, y, du, p)
                return du

            if NUMBA_DISABLE_JIT:
                address = id(differential_njit)
            else:
                address = differential_numbalsoda.address
            self._cache_odeint[name] = differential_odeint
            self._cache_odeint[address] = differential_odeint
            self._cache_solve_ivp[name] = differential_solve_ivp
            self._cache_solve_ivp[address] = differential_solve_ivp
            return address

    def get_cfunc(self, key: DifferentialPointer) -> Differential:
        with self._lock:
            return self._cache_cfunc[key]

    def get_odeint(self, key: DifferentialPointer) -> DifferentialOdeint:
        with self._lock:
            return self._cache_odeint[key]

    def get_solve_ivp(self, key: DifferentialPointer) -> DifferentialSolveIVP:
        with self._lock:
            return self._cache_solve_ivp[key]
