"""Workarounds for bugs in Numba.

These are applied when :mod:`pttools.speedup` is imported,
which happens before any PTtools function is compiled.
"""

# The private members of Numba have to be accessed for patching them.
# ruff: noqa: SLF001

import logging
import typing as tp

from numba.core.compiler import CompileResult

logger = logging.getLogger(__name__)

_REBUILD_ORIG: tp.Callable[..., CompileResult] = CompileResult._rebuild.__func__  # type: ignore[attr-defined]


def _rebuild_with_reload_init(cls: type[CompileResult], *args, **kwargs) -> CompileResult:
    """Rebuild a compile result from the Numba cache and keep its ``reload_init`` on the loaded library.

    A cached parallel function needs the Numba threading layer to be launched before its machine code
    can be loaded, since the code refers to the symbols of the threading layer.
    Numba tracks this with ``reload_init``, a list of functions that are called before loading from the cache.
    When a function is compiled, it inherits the ``reload_init`` of the functions it calls,
    since their code is linked into its own.
    However, Numba does not restore ``reload_init`` on a library that has been loaded from the cache.
    A function that is compiled in a process that loaded a parallel callee from the cache
    is therefore cached with an empty ``reload_init``,
    even though its machine code contains the parallel code of the callee.
    When another process then loads that function from the cache before it has launched the threading layer,
    the symbols of the threading layer are unresolved.
    On Linux this results in a segmentation fault when the function is called,
    and on macOS LLVM aborts the process while loading the code.

    Numba fixed the propagation of ``reload_init`` for freshly compiled callees in
    `numba/numba#9950 <https://github.com/numba/numba/pull/9950>`_,
    but callees loaded from the cache are still missed as of Numba 0.66.
    """
    cres: CompileResult = _REBUILD_ORIG(cls, *args, **kwargs)
    if cres.reload_init:
        cres.library._reload_init.update(cres.reload_init)
    return cres


def patch_reload_init() -> None:
    """Apply :func:`_rebuild_with_reload_init` to Numba."""
    if CompileResult._rebuild.__func__ is _rebuild_with_reload_init:  # type: ignore[attr-defined]
        return
    logger.debug("Patching Numba to keep reload_init on libraries loaded from the cache.")
    CompileResult._rebuild = classmethod(_rebuild_with_reload_init)  # type: ignore[method-assign]


patch_reload_init()
