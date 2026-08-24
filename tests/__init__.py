"""
Unit tests for PTtools.

These tests are not part of PTtools itself and are not included in the PTtools packages such as wheels.
However, they contain good examples on how to use PTtools and its dependencies.

Test parallelisation
--------------------
The tests are run in parallel with pytest-xdist using ``--dist=loadgroup``.
The ``pytest_collection_modifyitems`` hook in ``conftest.py`` adds the ``xdist_group`` marker automatically
to the tests of those classes that have class-level state,
which is detected from the presence of a ``setUpClass``, ``tearDownClass``, ``setup_class`` or ``teardown_class``
method in the class or in one of its base classes.
(The no-op implementations that are inherited from :class:`unittest.TestCase` are ignored.)
The tests of the other classes are distributed freely across the workers.

This can be overridden in two ways.

- Set the ``RUN_ON_SAME_WORKER`` class attribute to ``True`` or ``False``
  to force all tests of that class to be run on the same worker or to allow them to be distributed.
- Apply the ``@pytest.mark.xdist_group("group_name")`` marker to individual tests or to a whole class
  to run tests from different classes on the same worker.
  Manually applied markers are never overridden by the hook.
"""

from pttools.logging import setup_logging


setup_logging()
