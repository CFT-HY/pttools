"""Unit tests for the utilities of the documentation examples."""

import importlib.util
import sys
import types
import unittest
from unittest import mock

from pttools.docs.examples import is_sphinx_gallery


def fake_main() -> types.ModuleType:
    """A ``__main__`` module without ``__file__``, like the one in which Sphinx-Gallery executes the examples."""
    spec = importlib.util.spec_from_loader("__main__", None)
    assert spec is not None
    return importlib.util.module_from_spec(spec)


class IsSphinxGalleryTest(unittest.TestCase):
    """Tests for detecting whether an example is executed by Sphinx-Gallery."""

    def test_sphinx_gallery(self) -> None:
        """Sphinx-Gallery is detected when it has been imported and __main__ has no __file__."""
        modules = {"sphinx_gallery": types.ModuleType("sphinx_gallery"), "__main__": fake_main()}
        with mock.patch.dict(sys.modules, modules):
            assert is_sphinx_gallery()

    def test_script(self) -> None:
        """A script that has a file is not detected as a Sphinx-Gallery example."""
        main = types.ModuleType("__main__")
        main.__file__ = "example.py"
        with mock.patch.dict(sys.modules, {"sphinx_gallery": types.ModuleType("sphinx_gallery"), "__main__": main}):
            assert not is_sphinx_gallery()

    def test_without_sphinx_gallery(self) -> None:
        """Without Sphinx-Gallery, e.g. in an interactive session, the example is not detected as one."""
        with mock.patch.dict(sys.modules, {"__main__": fake_main()}):
            sys.modules.pop("sphinx_gallery", None)
            assert not is_sphinx_gallery()


if __name__ == "__main__":
    unittest.main()
