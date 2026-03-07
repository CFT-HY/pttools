import threading

import pttools.type_hints as th

CS2_CACHE: dict[th.CS2FunScalarPtr, th.CS2CFuncType] = {}
CS2_CACHE_LOCK = threading.Lock()


def cs2_converter(cs2_fun_ptr: th.CS2FunScalarPtr) -> th.CS2CFuncType:
    r"""Converter for getting a $c_s^2$ ctypes function from a pointer

    This is a rather ugly hack. There should be a better way to call a function by a pointer!
    """
    with CS2_CACHE_LOCK:
        if cs2_fun_ptr in CS2_CACHE:
            return CS2_CACHE[cs2_fun_ptr]
        # https://numba.pydata.org/numba-doc/0.15.1/interface_c.html
        cs2_fun = th.CS2CFunc(cs2_fun_ptr)
        CS2_CACHE[cs2_fun_ptr] = cs2_fun
        return cs2_fun
