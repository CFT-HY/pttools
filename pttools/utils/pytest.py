"""Utilities for the pytest test suites of PTtools and the projects that use it, such as PTPlot.

The hooks and fixtures of this module are found by pytest when they are imported in ``conftest.py``, e.g.

.. code-block:: python

    from pttools.utils.pytest import (
        log_test_name_at_start,
        pytest_collection_modifyitems,
        pytest_configure,
        pytest_sessionfinish,
        pytest_xdist_auto_num_workers,
    )

This module requires pytest, which is not a runtime dependency of PTtools,
and is therefore not imported by :mod:`pttools.utils`.
"""

import logging
import os
import time

import matplotlib
import pytest

from pttools.logging import setup_logging
from pttools.utils import system
from pttools.utils.system import AVAILABLE_CPU_CORES

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "CLASS_SETUP_METHODS",
    "MAX_XDIST_WORKERS",
    "RUN_ON_SAME_WORKER_ATTR",
    "XDIST_GROUP_MARKER",
    "configure_testing",
    "defines_class_setup",
    "log_test_name_at_start",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_sessionfinish",
    "pytest_xdist_auto_num_workers",
    "run_on_same_worker",
    "xdist_group_name",
]

#: Class-level setup and teardown methods.
#: If a test class or one of its base classes defines any of these,
#: then it presumably has state that is shared between its test methods,
#: and therefore all its tests have to be run on the same worker.
CLASS_SETUP_METHODS: tuple[str, ...] = ("setUpClass", "tearDownClass", "setup_class", "teardown_class")

#: Maximum number of pytest-xdist workers for ``--numprocesses=auto``.
MAX_XDIST_WORKERS: int = 8

#: Name of the class attribute with which a test class can manually configure
#: whether all of its tests should be run on the same pytest-xdist worker.
RUN_ON_SAME_WORKER_ATTR: str = "RUN_ON_SAME_WORKER"

#: Name of the pytest-xdist marker that groups tests to the same worker.
#: This is used by ``--dist=loadgroup``.
XDIST_GROUP_MARKER: str = "xdist_group"


def configure_testing(name: str = "pttools", log_dir: str | os.PathLike[str] | None = None) -> None:
    """Mark that unit tests are running, select the non-interactive Matplotlib backend and set up logging.

    Without a display Matplotlib falls back to the Agg backend automatically,
    but with one (e.g. ``$DISPLAY`` of SSH X11 forwarding) it would select an interactive backend such as TkAgg.
    The objects of Tkinter must not be used outside the main thread,
    but their ``__del__`` methods can be called by the garbage collector in any thread,
    e.g. in the worker threads of :func:`pttools.speedup.parallel.run_parallel`,
    which can deadlock the test.

    This is called by :func:`pytest_configure` with the default arguments.
    Other projects can call this in their own ``pytest_configure`` with their own arguments.

    :param name: name of the log files, see :func:`pttools.logging.setup_logging`
    :param log_dir: directory of the log files, see :func:`pttools.logging.setup_logging`
    """
    matplotlib.use("Agg", force=True)
    system._TESTING = True  # noqa: SLF001
    setup_logging(name=name, log_dir=log_dir)


def defines_class_setup(cls: type) -> bool:
    """Whether the test class or one of its base classes defines class-level setup or teardown.

    The implementations that are inherited from :class:`unittest.TestCase` are ignored,
    as those are no-ops and therefore don't create any state that would have to be shared between the tests.
    """
    for klass in cls.__mro__:
        module: str = getattr(klass, "__module__", "")
        if module == "unittest" or module.startswith("unittest."):
            continue
        if any(name in vars(klass) for name in CLASS_SETUP_METHODS):
            return True
    return False


@pytest.fixture(autouse=True)
def log_test_name_at_start(request: pytest.FixtureRequest) -> None:
    """
    Before starting a test, log its name.

    This makes it easier to retrieve the logs for a specific test.
    """
    logger.info("=" * 20 + request.node.nodeid + "=" * 20)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Group the tests that have to be run on the same pytest-xdist worker.

    With ``--dist=loadgroup`` the tests that share an ``xdist_group`` marker are run on the same worker,
    and the rest of the tests are distributed individually.
    Here the marker is added automatically to the tests of those classes that have class-level state.
    Tests and classes that already have the marker are left as they are,
    which makes it possible to group specific tests from different classes to the same worker.

    This hook has to be run before the corresponding hook of pytest-xdist,
    which appends the group names to the node IDs of the tests.
    """
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is None:
            # Tests that are not in a class have no class-level state to share.
            continue
        if item.get_closest_marker(XDIST_GROUP_MARKER) is not None:
            # Don't override manually specified groups.
            continue
        if not run_on_same_worker(cls):
            continue
        item.add_marker(pytest.mark.xdist_group(xdist_group_name(item, cls)))


def pytest_configure(config: pytest.Config) -> None:
    """Configure the test suite of PTtools with :func:`configure_testing`."""
    configure_testing()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Log the time at which the test run finished.

    With pytest-xdist this is logged by both the controller and each worker,
    which have log files of their own.
    """
    worker_id: str | None = getattr(session.config, "workerinput", {}).get("workerid")
    logger.info(
        "Test run finished at %s%s with exit status %s.",
        time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "" if worker_id is None else f" on worker {worker_id}",
        exitstatus
    )


def pytest_xdist_auto_num_workers() -> int | None:
    """Number of pytest-xdist workers to use for ``--numprocesses=auto``.

    :return: Number of workers (None = auto)
    """
    if AVAILABLE_CPU_CORES is not None and AVAILABLE_CPU_CORES >= MAX_XDIST_WORKERS:
        return MAX_XDIST_WORKERS
    return None


def run_on_same_worker(cls: type) -> bool:
    """Whether all tests of the given test class should be run on the same pytest-xdist worker.

    This can be configured manually by setting the ``RUN_ON_SAME_WORKER`` class attribute.
    Otherwise, it's deduced from the presence of class-level setup or teardown methods.
    """
    manual = getattr(cls, RUN_ON_SAME_WORKER_ATTR, None)
    if manual is not None:
        return bool(manual)
    return defines_class_setup(cls)


def xdist_group_name(item: pytest.Item, cls: type) -> str:
    """Name of the pytest-xdist group for the test class of the given test.

    The node ID of the class is used, so that identically named classes in different modules
    don't end up in the same group.
    """
    parent = item.getparent(pytest.Class)
    if parent is not None:
        # The group name is separated from the node ID with "@", and must therefore not contain it.
        return parent.nodeid.replace("@", "_")
    return f"{cls.__module__}.{cls.__qualname__}".replace("@", "_")
