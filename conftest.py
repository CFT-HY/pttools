import logging
# import typing as tp

import pytest

from pttools.logging import setup_logging

# if tp.TYPE_CHECKING:
#     from _pytest.fixtures import SubRequest

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config):
    setup_logging()


@pytest.fixture(scope="function", autouse=True)
def log_test_name_at_start(request):
    """
    Before starting a test, log its name.
    This makes it easier to retrieve the logs for a specific test.
    """
    logger.info("=" * 20 + request.node.nodeid + "=" * 20)
