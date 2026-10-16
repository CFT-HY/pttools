"""Tests for the generic decorators."""

import inspect
import typing as tp
import unittest

from pttools.utils.decorators import cached_property


class Base:
    """A base class with documented attributes."""

    @property
    def documented(self) -> float:
        """Documented in the base class."""
        return 1.

    @property
    def own_doc(self) -> float:
        """Documented in the base class."""
        return 1.


class Child(Base):
    """A subclass that overrides the attributes with cached properties."""

    def __init__(self) -> None:
        """Initialise the call counter."""
        self.calls: int = 0

    @cached_property
    @tp.override
    def documented(self) -> float:
        self.calls += 1
        return 2.

    @cached_property
    @tp.override
    def own_doc(self) -> float:
        """Documented in the subclass."""
        return 2.

    @cached_property
    def undocumented(self) -> float:  # noqa: D102
        return 3.


class CachedPropertyTest(unittest.TestCase):
    """Tests for the cached property that inherits docstrings."""

    @staticmethod
    def test_inherited_docstring() -> None:
        """A cached property without a docstring should inherit the docstring of the overridden attribute."""
        assert inspect.getdoc(inspect.getattr_static(Child, "documented")) == "Documented in the base class."

    @staticmethod
    def test_own_docstring() -> None:
        """A docstring of the cached property itself should take precedence."""
        assert inspect.getdoc(inspect.getattr_static(Child, "own_doc")) == "Documented in the subclass."

    @staticmethod
    def test_no_docstring() -> None:
        """A cached property without a docstring in the class hierarchy should have no docstring."""
        assert inspect.getdoc(inspect.getattr_static(Child, "undocumented")) is None

    @staticmethod
    def test_caching() -> None:
        """The value should be computed only once."""
        child = Child()
        assert child.documented == 2.
        assert child.documented == 2.
        assert child.calls == 1


if __name__ == "__main__":
    unittest.main()
