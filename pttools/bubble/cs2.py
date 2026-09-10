r"""Calling $c_s^2$ functions by their pointers.

Jitted functions should take the $c_s^2$ function as a pointer instead of as a function object,
since the Numba type of a function object is tied to the identity of the ``CPUDispatcher`` object,
which is created anew in every process.
Such an argument would therefore result in a different cache key on every run,
which would prevent the Numba on-disk cache from ever hitting.
See the "Numba caching" section of the developer documentation for the details.
"""

import threading

from llvmlite import ir
import numba
from numba.extending import intrinsic, overload

from pttools.speedup.numba_wrapper import CFunc, Dispatcher
from pttools.speedup.options import NUMBA_DISABLE_JIT
import pttools.type_hints as th

CS2_CFUNCS: dict[th.CS2FunScalarPtr, CFunc] = {}
r"""Compiled $c_s^2$ functions by their pointers.

These references keep the compiled functions alive, which is necessary for keeping their pointers valid.
"""

CS2_FUNCS: dict[th.CS2FunScalarPtr, th.CS2Fun | th.CS2CFuncType] = {}
r"""Python-callable versions of the $c_s^2$ functions by their pointers.

These are used when calling :func:`cs2_from_ptr` outside jitted code.
For a compiled function this is a ctypes function, and otherwise the function itself.
"""

CS2_FUNCS_LOCK = threading.Lock()


def cs2_to_ptr(cs2_fun: th.CS2Fun) -> th.CS2FunScalarPtr:
    r"""Get a pointer for calling the given $c_s^2$ function from jitted code.

    The function is compiled to a :func:`numba.cfunc`, and the address of that cfunc is returned.
    If Numba jitting is disabled, nothing is compiled, and therefore ``id()`` of the function
    is used as its pointer instead, in the same way as
    :class:`pttools.speedup.differential.DifferentialCache` does for the differentials.

    The pointer is valid only in the process in which it was created,
    and in the processes forked from it.
    Therefore, it must not be baked into cached Numba functions, e.g. as a default argument value.

    :param cs2_fun: $c_s^2$ function, which has to be callable with scalar arguments
    :return: pointer to the $c_s^2$ function
    """
    if isinstance(cs2_fun, CFunc):
        cfunc = cs2_fun
    elif NUMBA_DISABLE_JIT:
        # Jitting is disabled, and therefore the function cannot be compiled to a cfunc.
        # Nothing is compiled in this case, and therefore the function is called by a Python-level lookup.
        ptr = id(cs2_fun)
        with CS2_FUNCS_LOCK:
            CS2_FUNCS[ptr] = cs2_fun
        return ptr
    elif isinstance(cs2_fun, Dispatcher):
        cs2_jit = cs2_fun

        # Caching is disabled, as these functions are created dynamically.
        @numba.cfunc(th.CS2FunScalarSig, cache=False)
        def cs2_cfunc(w: float, phase: float) -> float:
            return cs2_jit(w, phase)

        cfunc = cs2_cfunc
    else:
        cfunc = numba.cfunc(th.CS2FunScalarSig, cache=False)(cs2_fun)

    ptr = cfunc.address
    with CS2_FUNCS_LOCK:
        CS2_CFUNCS[ptr] = cfunc
        # The ctypes version is used instead of the CFunc object itself,
        # since calling a CFunc object from Python calls the uncompiled function.
        CS2_FUNCS[ptr] = cfunc.ctypes
    return ptr


@intrinsic
def _cs2_from_ptr_intrinsic(typingctx, cs2_ptr, w, phase):
    r"""Call a $c_s^2$ cfunc by its address.

    Numba has no built-in way of converting an integer to a function pointer,
    and therefore the call is generated directly as LLVM IR.
    The resulting Numba signature contains only integers and floats,
    which makes the calling function cacheable.
    """
    if not isinstance(cs2_ptr, numba.types.Integer):
        raise TypeError(f"The cs2 pointer must be an integer. Got: {cs2_ptr}")
    if not isinstance(w, numba.types.Number) or not isinstance(phase, numba.types.Number):
        raise TypeError(f"The arguments of cs2 must be numbers. Got: w={w}, phase={phase}")
    sig = numba.types.float64(numba.types.uintp, numba.types.float64, numba.types.float64)

    def codegen(context, builder, signature, args):
        ptr, w_arg, phase_arg = args
        fun_type = ir.FunctionType(ir.DoubleType(), (ir.DoubleType(), ir.DoubleType()))
        fun_ptr = builder.inttoptr(ptr, fun_type.as_pointer())
        return builder.call(fun_ptr, (w_arg, phase_arg))

    return sig, codegen


def cs2_from_ptr(cs2_ptr: th.CS2FunScalarPtr, w: float, phase: float) -> float:
    r"""Compute $c_s^2(w,\phi)$ with the $c_s^2$ function that the given pointer points to.

    :param cs2_ptr: pointer to the $c_s^2$ function, as returned by :func:`cs2_to_ptr`
    :param w: enthalpy $w$
    :param phase: phase $\phi$
    :return: speed of sound squared $c_s^2$
    """
    cs2_fun = CS2_FUNCS.get(cs2_ptr)
    if cs2_fun is None:
        # The pointer was not created by cs2_to_ptr, and is therefore presumed to be the address of a cfunc.
        # https://numba.pydata.org/numba-doc/0.15.1/interface_c.html
        cs2_fun = th.CS2CFunc(cs2_ptr)
        with CS2_FUNCS_LOCK:
            CS2_FUNCS[cs2_ptr] = cs2_fun
    return float(cs2_fun(w, phase))


def _cs2_from_ptr_impl(cs2_ptr: th.CS2FunScalarPtr, w: float, phase: float) -> float:
    """A non-jitted wrapper for :py:func:_cs2_from_ptr_intrinsic:.

    This function exists, because functions decorated with
    :py:func:`numba.extending.overload` have to return regular Python functions.
    """
    return _cs2_from_ptr_intrinsic(cs2_ptr, w, phase)


@overload(cs2_from_ptr, jit_options={"nopython": True})
def _cs2_from_ptr_numba(cs2_ptr: th.CS2FunScalarPtr, w: float, phase: float) -> th.NumbaFunc:
    return _cs2_from_ptr_impl
