"""Utilities for handling functions for the differential equations"""

import logging
import threading
import typing as tp

import numba
import numpy as np

from pttools.speedup.numba_wrapper import CFunc, CPUDispatcher, lsoda_sig
from pttools.speedup.options import NUMBA_DISABLE_JIT
import pttools.type_hints as th

logger = logging.getLogger(__name__)

type DifferentialCFunc = tp.Callable[[float, th.FloatArr1D, th.FloatArr1D, th.FloatArr1D | None], None] | CFunc
type DifferentialOdeint = tp.Callable[[th.FloatArr1D, float, th.FloatArr1D | None], th.FloatArr1D] | CPUDispatcher
type DifferentialSolveIVP = tp.Callable[[float, th.FloatArr1D, th.FloatArr1D | None], th.FloatArr1D] | CPUDispatcher
type Differential = DifferentialCFunc | DifferentialOdeint | DifferentialSolveIVP
type DifferentialPointer = numba.types.CPointer  # CPointer(lsoda_sig)
type DifferentialKey = DifferentialPointer | str


class DifferentialCache:
    """Cache for the functions that compute the differentials

    This cache system automatically compiles versions for
    :func:`scipy.integrate.odeint`,
    :func:`scipy.integrate.solve_ivp`
    and NumbaLSODA.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cache_njit: dict[DifferentialKey, DifferentialCFunc] = {}
        self._cache_odeint: dict[DifferentialKey, DifferentialOdeint] = {}
        self._cache_pointers: dict[str, DifferentialPointer] = {}
        self._cache_solve_ivp: dict[DifferentialKey, DifferentialSolveIVP] = {}

    def __contains__(self, item: DifferentialKey) -> bool:
        return item in self._cache_njit

    def add(
            self,
            name: str,
            differential: DifferentialCFunc,
            p_last_is_backwards: bool = True,
            ndim: int = 3) -> DifferentialPointer:
        """Add a differential function to the cache"""
        with self._lock:
            if name in self._cache_njit:
                logger.warning(
                    "Attempted to add a differential with the name \"%s\" which is already in the cache. "
                    "This may be caused by multiprocessing giving the same id to a different object in a different process. "
                    "Creating a new differential. This will ensure that the new differential is correct, "
                    "and it will not affect access to the old differential using its pointer.",
                    name
                )
            differential_njit = numba.njit(differential)
            if not NUMBA_DISABLE_JIT:
                differential_cfunc = numba.cfunc(lsoda_sig)(differential)
                if p_last_is_backwards:
                    @numba.cfunc(lsoda_sig)
                    def differential_numbalsoda(
                            t: float,
                            u: th.FloatArr1D,
                            du: th.FloatArr1D,
                            p: th.FloatArr1D) -> None:
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
            def differential_odeint(y: th.FloatArr1D, t: float, p: th.FloatArr1D | None = None) -> th.FloatArr1D:
                du = np.empty_like(y)
                differential_njit(t, y, du, p)
                return du

            @numba.njit
            def differential_solve_ivp(t: float, y: th.FloatArr1D, p: th.FloatArr1D | None = None) -> th.FloatArr1D:
                du = np.empty_like(y)
                differential_njit(t, y, du, p)
                return du

            if NUMBA_DISABLE_JIT:
                address = id(differential_njit)
            else:
                address = differential_numbalsoda.address  # pylint: disable=possibly-used-before-assignment
            self._cache_pointers[name] = address

            self._cache_njit[address] = differential_njit
            self._cache_odeint[address] = differential_odeint
            self._cache_solve_ivp[address] = differential_solve_ivp

            self._cache_njit[name] = differential_njit
            self._cache_odeint[name] = differential_odeint
            self._cache_solve_ivp[name] = differential_solve_ivp
        return address

    def _get_func(self, key: DifferentialKey, cache: dict[DifferentialKey, Differential]) -> Differential:
        try:
            with self._lock:
                return cache[key]
        except KeyError as error:
            raise KeyError(
                f"Could not find differential function in the cache with the key \"{key}\". "
                f"This may indicate an issue with parallelism. Available functions: {cache.keys()}") from error

    def get_njit(self, key: DifferentialKey) -> DifferentialCFunc:
        """Get a Numba-jitted function"""
        return self._get_func(key, self._cache_njit)

    def get_odeint(self, key: DifferentialKey) -> DifferentialOdeint:
        """Get a function compatible with SciPy odeint"""
        return self._get_func(key, self._cache_odeint)

    def get_pointer(self, name: str) -> DifferentialPointer:
        """Get a pointer to the function from its name"""
        return self._cache_pointers[name]

    def get_solve_ivp(self, key: DifferentialKey) -> DifferentialSolveIVP:
        """Get a function compatible with SciPy solve_ivp"""
        return self._get_func(key, self._cache_solve_ivp)

    def keys(self):
        """Get the keys in the cache"""
        return self._cache_njit.keys()

    @property
    def size(self) -> int:
        """Get the number of differentials in the cache"""
        with self._lock:
            return len(self._cache_pointers)
