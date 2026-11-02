"""Locking for the thread-unsafe parts of Matplotlib.

Matplotlib is not thread-safe.
Both the global ``rcParams``, which are modified by :func:`matplotlib.rc_context`,
and the mathtext parser, which is shared by all figures, are mutated when figures are created and rendered.
When figures are created or rendered in parallel threads,
e.g. by a web server such as that of PTPlot,
whose pages contain several figures that the browser requests simultaneously,
this corrupts the shared state of Matplotlib and results in random errors such as::

    ValueError: ParseException: exception raised in parse action (at char 0), (line:1, col:1)

Therefore, the creation and rendering of the figures has to be serialised with the lock of this module.
The global figure registry of :mod:`matplotlib.pyplot` is not thread-safe either,
and therefore :func:`pttools.plot.utils.show_fig` and :func:`pttools.plot.utils.show_figs` also use the lock.
"""

import functools
import threading
import typing as tp

__all__ = ["MATPLOTLIB_LOCK", "matplotlib_lock"]

#: Global lock for Matplotlib.
#: This is re-entrant so that the locked functions can call each other.
MATPLOTLIB_LOCK: threading.RLock = threading.RLock()


def matplotlib_lock[**P, R](func: tp.Callable[P, R]) -> tp.Callable[P, R]:
    """Serialise the calls of the decorated function with the :data:`MATPLOTLIB_LOCK`."""
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with MATPLOTLIB_LOCK:
            return func(*args, **kwargs)
    return wrapper
