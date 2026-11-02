"""Configuration for the pytest test suite.

The hooks and fixtures are defined in :mod:`pttools.utils.pytest`, so that they can be used by PTPlot as well.
Pytest finds them, as they are imported here.
"""

import pytest

from pttools.utils.pytest import (
    log_test_name_at_start,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_sessionfinish,
    pytest_xdist_auto_num_workers,
)

# The imports are used by pytest.
__all__ = [
    "log_test_name_at_start",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_sessionfinish",
    "pytest_xdist_auto_num_workers",
]

# Pytest rewrites the assertions only in the test modules and conftest.py by default.
# The base classes of the tests are in other modules, so that they are not collected as tests themselves.
pytest.register_assert_rewrite("tests.models.base_bag", "tests.models.base_model")
