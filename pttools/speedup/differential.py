"""Utilities for handling functions for the differential equations"""

import logging
import threading
import typing as tp

import numba
import numpy as np

from pttools.speedup.numba_wrapper import CFunc, CPUDispatcher, lsoda_sig
from pttools.speedup.options import NUMBA_DISABLE_JIT

logger = logging.getLogger(__name__)

DifferentialCFunc = tp.Union[tp.Callable[[float, np.ndarray, np.ndarray, tp.Optional[np.ndarray]], None], CFunc]
DifferentialOdeint = tp.Union[tp.Callable[[np.ndarray, float, tp.Optional[np.ndarray]], np.ndarray], CPUDispatcher]
DifferentialSolveIVP = tp.Union[tp.Callable[[float, np.ndarray, tp.Optional[np.ndarray]], np.ndarray], CPUDispatcher]
Differential = tp.Union[DifferentialCFunc, DifferentialOdeint, DifferentialSolveIVP]
DifferentialPointer = numba.types.CPointer(lsoda_sig)
DifferentialKey = tp.Union[DifferentialPointer, str]


class DifferentialCache:
    """Cache for the functions that compute the differentials

    This cache system automatically compiles versions for
    :func:`scipy.integrate.odeint`,
    :func:`scipy.integrate.solve_ivp`
    and NumbaLSODA.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cache_njit: tp.Dict[DifferentialKey, DifferentialCFunc] = {}
        self._cache_odeint: tp.Dict[DifferentialKey, DifferentialOdeint] = {}
        self._cache_pointers: tp.Dict[str, DifferentialPointer] = {}
        self._cache_solve_ivp: tp.Dict[DifferentialKey, DifferentialSolveIVP] = {}

    def __contains__(self, item: DifferentialKey) -> bool:
        return item in self._cache_njit

    def add(
            self,
            name: str, differential: DifferentialCFunc,
            p_last_is_backwards: bool = True,
            ndim: int = 3) -> DifferentialPointer:
        with self._lock:
            if name in self._cache_njit:
                raise ValueError("The key is already in the cache")
            differential_njit = numba.njit(differential)
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
            self._cache_pointers[name] = address

            self._cache_njit[address] = differential_njit
            self._cache_odeint[address] = differential_odeint
            self._cache_solve_ivp[address] = differential_solve_ivp

            self._cache_njit[name] = differential_njit
            self._cache_odeint[name] = differential_odeint
            self._cache_solve_ivp[name] = differential_solve_ivp
        return address

    def _get_func(self, key: DifferentialKey, cache: tp.Dict[DifferentialKey, Differential]) -> Differential:
        try:
            with self._lock:
                return cache[key]
        except KeyError as error:
            raise KeyError(
                f"Could not find differential function in the cache with the key \"{key}\". "
                f"This may indicate an issue with parallelism. Available functions: {cache.keys()}") from error

    def get_njit(self, key: DifferentialKey) -> DifferentialCFunc:
        return self._get_func(key, self._cache_njit)

    def get_odeint(self, key: DifferentialKey) -> DifferentialOdeint:
        return self._get_func(key, self._cache_odeint)

    def get_pointer(self, name: str) -> DifferentialPointer:
        return self._cache_pointers[name]

    def get_solve_ivp(self, key: DifferentialKey) -> DifferentialSolveIVP:
        return self._get_func(key, self._cache_solve_ivp)

    def keys(self):
        return self._cache_njit.keys()
