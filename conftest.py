"""Configuration for the pytest test suite"""

import logging
# import typing as tp

import pytest

from pttools.logging import setup_logging
from pttools.utils.system import AVAILABLE_CPU_CORES

# if tp.TYPE_CHECKING:
#     from _pytest.fixtures import SubRequest

logger = logging.getLogger(__name__)

#: Name of the class attribute with which a test class can manually configure
#: whether all of its tests should be run on the same pytest-xdist worker.
RUN_ON_SAME_WORKER_ATTR: str = "RUN_ON_SAME_WORKER"

#: Name of the pytest-xdist marker that groups tests to the same worker.
#: This is used by ``--dist=loadgroup``.
XDIST_GROUP_MARKER: str = "xdist_group"

#: Class-level setup and teardown methods.
#: If a test class or one of its base classes defines any of these,
#: then it presumably has state that is shared between its test methods,
#: and therefore all its tests have to be run on the same worker.
CLASS_SETUP_METHODS: tuple[str, ...] = ("setUpClass", "tearDownClass", "setup_class", "teardown_class")


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


def pytest_xdist_auto_num_workers() -> int | None:
    """Number of pytest-xdist workers to use for ``--numprocesses=auto``

    :return: Number of workers (None = auto)
    """
    if AVAILABLE_CPU_CORES >= 8:
        return 8
    return None


def pytest_configure(config: pytest.Config) -> None:
    setup_logging()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]):
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


@pytest.fixture(scope="function", autouse=True)
def log_test_name_at_start(request):
    """
    Before starting a test, log its name.
    This makes it easier to retrieve the logs for a specific test.
    """
    logger.info("=" * 20 + request.node.nodeid + "=" * 20)
