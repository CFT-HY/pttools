"""Tests for documenting the attributes with the descriptions of the exportable fields."""

import importlib
import importlib.util
import io
from pathlib import Path
import typing as tp
import unittest
from unittest import mock

from pttools.bubble import Bubble
from pttools.docs.fields import (
    FIELD_DEFINITION_MODULES,
    FIELD_MODULES,
    add_field_docs,
    attribute_name,
    field_attribute_docs,
    format_description,
    note_field_dependencies,
)
from pttools.utils.fields import Extractable, Field

HAS_SPHINX: bool = importlib.util.find_spec("sphinx") is not None


def all_descriptions() -> dict[str, str]:
    """All the field descriptions, with the class and field names where they are used."""
    for module in FIELD_MODULES:
        importlib.import_module(module)
    descriptions: dict[str, str] = {}
    stack: list[type] = [Extractable]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        for field in cls.FIELDS.values():
            if field.description:
                descriptions.setdefault(field.description, f"{cls.__name__}.{field.name}")
    return descriptions


class AttributeNameTest(unittest.TestCase):
    """Tests for finding the attribute of a field."""

    def test_attribute(self) -> None:
        """Test that the attribute name is the field name, or the getter if it is an attribute name."""
        assert attribute_name(Field("v_wall")) == "v_wall"
        assert attribute_name(Field("thin_shell_limit", getter="thin_shell_t_points_min")) == "thin_shell_t_points_min"

    def test_not_attribute(self) -> None:
        """Test that fields with a dotted or callable getter, or with call or index, have no attribute."""
        assert attribute_name(Field("v_wall", getter="bubble.v_wall")) is None
        assert attribute_name(Field("omgw0_h2", call=True)) is None
        assert attribute_name(Field("snr", call=True, index=0)) is None
        assert attribute_name(Field("x", getter=abs)) is None


class FormatDescriptionTest(unittest.TestCase):
    """Tests for formatting the field descriptions as attribute documentation."""

    def test_capitalize(self) -> None:
        """Test that the first letter of the description is capitalized."""
        assert format_description("name of the model") == "Name of the model"

    def test_unchanged(self) -> None:
        """Test that descriptions starting with math or an uppercase letter are not changed."""
        assert format_description(r"$v_\text{wall}$, wall speed") == r"$v_\text{wall}$, wall speed"
        assert format_description("LaTeX label") == "LaTeX label"


@unittest.skipUnless(HAS_SPHINX, "Sphinx is not installed")
class FieldAttributeDocsTest(unittest.TestCase):
    """Tests for the attribute documentation from the field descriptions."""

    docs: dict[tuple[str, str, str], str]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.docs = field_attribute_docs()

    def test_attribute(self) -> None:
        """Test that an attribute is documented with the description of its field."""
        assert self.docs["pttools.bubble.bubble.base", "BaseBubble", "v_wall"] == r"$v_\text{wall}$, wall speed"

    def test_most_basic_class(self) -> None:
        """An attribute should be documented in the class that introduces it, not in the subclasses."""
        assert ("pttools.models.base", "BaseModel", "label_latex") in self.docs
        assert ("pttools.models.bag", "BagModel", "label_latex") not in self.docs

    def test_fallback_to_field_class(self) -> None:
        """Attributes that the source code analysis does not find should be documented in the class of the field."""
        assert ("pttools.models.model", "Model", "w_min") in self.docs

    def test_no_properties(self) -> None:
        """Properties and methods have docstrings of their own."""
        names = {attr for _, _, attr in self.docs}
        for name in ("css2_Tn", "kappa", "pow_gw", "omgw0_h2", "f_max"):
            assert name not in names

    def test_no_attributes_of_other_objects(self) -> None:
        """Fields that get attributes of other objects should not be documented as attributes of the class."""
        assert ("pttools.ssm.spectrum", "SSMSpectrum", "v_wall") not in self.docs

    def test_add_field_docs(self) -> None:
        """Test that the field docs are added to the Sphinx module analyzer, keeping the existing comments."""
        from sphinx.pycode import ModuleAnalyzer  # noqa: PLC0415
        add_field_docs()
        analyzer = ModuleAnalyzer.for_module("pttools.bubble.bubble.base")
        assert analyzer.attr_docs["BaseBubble", "v_wall"] == [r"$v_\text{wall}$, wall speed", ""]
        # Existing comments are kept
        assert analyzer.attr_docs["BaseBubble", "solved"][0].startswith("Whether the solver provided")
        # Adding the docs again should not change anything.
        assert add_field_docs() == 0


class FieldDependenciesTest(unittest.TestCase):
    """Tests for the dependencies of the documents on the field definitions."""

    def test_extractable_class(self) -> None:
        """Test that the docs of an extractable class depend on the field definition modules."""
        app = mock.MagicMock()
        note_field_dependencies(app, "class", "Bubble", Bubble, None, [])
        paths = {Path(call.args[0]).name for call in app.env.note_dependency.call_args_list}
        assert app.env.note_dependency.call_count == len(FIELD_DEFINITION_MODULES)
        assert paths == {"export.py"}

    def test_other_objects(self) -> None:
        """Test that the docs of other classes and of attributes do not get field dependencies."""
        app = mock.MagicMock()
        note_field_dependencies(app, "class", "Path", Path, None, [])
        note_field_dependencies(app, "attribute", "Bubble.v_wall", None, None, [])
        app.env.note_dependency.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("docutils") is not None, "Docutils is not installed")
class DescriptionSyntaxTest(unittest.TestCase):
    """The field descriptions should be valid reStructuredText, as they are used in the documentation."""

    def test_descriptions(self) -> None:
        """Test that the descriptions of all the fields are parsed by docutils without warnings."""
        import docutils.core  # noqa: PLC0415
        for description, where in all_descriptions().items():
            with self.subTest(field=where):
                warnings = io.StringIO()
                docutils.core.publish_doctree(
                    format_description(description), settings_overrides={"warning_stream": warnings})
                assert warnings.getvalue() == ""


if __name__ == "__main__":
    unittest.main()
