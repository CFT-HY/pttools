"""Unit tests for the resolution of the cross-references whose target is documented with another name."""

import pytest

pytest.importorskip("sphinx", reason="Sphinx is not installed")

from pttools.docs.references import canonical_name, canonical_target

#: The class where the inherited attribute ``kappa`` of :py:class:`pttools.bubble.bubble.bubble.Bubble` is defined
KAPPA: str = "pttools.bubble.bubble.nucleated.BaseNucleatedBubble.kappa"


class TestCanonicalName:
    """Tests for :py:func:`pttools.docs.references.canonical_name`."""

    @staticmethod
    def test_canonical() -> None:
        """A name in the module where the object is defined is unchanged."""
        assert canonical_name("pttools.bubble.bubble.bubble.Bubble") == "pttools.bubble.bubble.bubble.Bubble"

    @staticmethod
    def test_inherited_attribute() -> None:
        """An inherited attribute is resolved to the class where it is defined."""
        assert canonical_name("pttools.bubble.bubble.Bubble.kappa") == KAPPA

    @staticmethod
    def test_missing() -> None:
        """A name that cannot be imported gives None."""
        assert canonical_name("pttools.bubble.bubble.Bubble.no_such_attribute") is None
        assert canonical_name("no_such_package.Foo") is None

    @staticmethod
    def test_module_attribute() -> None:
        """A module imported by another module is resolved to the module itself."""
        assert canonical_name("pttools.speedup.functions.np.gradient") == "numpy.gradient"

    @staticmethod
    def test_reexported() -> None:
        """A name re-exported by a package is resolved to the module where it is defined."""
        assert canonical_name("pttools.ssm.SSMSpectrum") == "pttools.ssm.spectrum.SSMSpectrum"


class TestCanonicalTarget:
    """Tests for :py:func:`pttools.docs.references.canonical_target`."""

    @staticmethod
    def test_class_context() -> None:
        """An unqualified name is looked up in the class of the docstring."""
        assert canonical_target("kappa", "pttools.bubble.bubble.bubble", "Bubble") == KAPPA

    @staticmethod
    def test_external_context() -> None:
        """A name in a module outside the resolved packages is not looked up."""
        assert canonical_target("Path", "pathlib", None) is None

    @staticmethod
    def test_module_context() -> None:
        """An unqualified name is looked up in the module of the docstring."""
        assert canonical_target("LatticeRun", "pttools.lattice", None) == "pttools.lattice.run.LatticeRun"
