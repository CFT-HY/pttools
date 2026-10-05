"""Tests for the field definitions and the extraction of the fields."""

import functools
import importlib
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import BUBBLE_FIELDS, Bubble
from pttools.models import BagModel, ConstCSModel
from pttools.omgw0 import SPECTRUM_FIELDS, SPECTRUM_FIELDS_F, SPECTRUM_FIELDS_Y, Spectrum
from pttools.ssm import SSM_SPECTRUM_FIELDS
from pttools.utils.fields import Extractable, Field, Fields, FieldShape, FieldType, Preset, describe, docstring_summary

#: The minimal parameters of the spectra
SPECTRUM_MINIMAL_PARAMS: tuple[str, ...] = (
    "v_wall", "alpha_n", "beta_tilde", "r_star", "T_star", "g_star", "cs2", "css2_Tn", "csb2_Tn")


def divmod_by_3(value: int) -> tuple[int, int]:
    """Test getter that returns a tuple."""
    return divmod(value, 3)


class Documented:
    """Test class with documented properties and methods."""

    #: A plain attribute
    attr: float = 1.

    @property
    def prop(self) -> float:
        r"""$p$, a property.

        More details.
        """
        return 1.

    @functools.cached_property
    def cached(self) -> float:
        """$c$, a cached property."""
        return 1.

    def method(self) -> float:
        """$m$, a method."""
        return 1.

    def pair(self) -> tuple[float, float]:
        """A method that returns a tuple."""
        return 1., 2.

    @staticmethod
    def static() -> float:
        """$s$, a static method."""
        return 1.

    # This is intentionally undocumented, as test_docstring_summary_none tests the handling of a missing docstring.
    def undocumented(self) -> float:  # noqa: D102
        return 1.


class DescribeTest(unittest.TestCase):
    """Tests for taking the descriptions of the fields from the docstrings."""

    def test_docstring_summary(self) -> None:
        """Test that the summary is the first line of the docstring without the trailing period."""
        assert docstring_summary(Documented, "prop") == "$p$, a property"
        assert docstring_summary(Documented, "cached") == "$c$, a cached property"
        assert docstring_summary(Documented, "method") == "$m$, a method"
        assert docstring_summary(Documented, "static") == "$s$, a static method"

    def test_docstring_summary_none(self) -> None:
        """Test that the summary is empty for attributes, undocumented methods and missing names."""
        assert docstring_summary(Documented, "attr") == ""
        assert docstring_summary(Documented, "undocumented") == ""
        assert docstring_summary(Documented, "missing") == ""

    def test_describe(self) -> None:
        """Test that a field without a description gets it from the docstring of its getter."""
        assert describe(Field("prop"), Documented).description == "$p$, a property"
        assert describe(Field("x", getter="method", call=True), Documented).description == "$m$, a method"

    def test_describe_keeps_explicit(self) -> None:
        """Test that a field with an explicit description is returned unchanged."""
        field = Field("prop", description="explicit")
        assert describe(field, Documented) is field

    def test_describe_not_applicable(self) -> None:
        """Test that fields whose getter is not a documented property or method are returned unchanged."""
        for field in (
                Field("attr"),
                Field("pair", call=True, index=0),
                Field("x", getter="prop.real"),
                Field("x", getter=abs)):
            with self.subTest(field=field.name):
                assert describe(field, Documented) is field

    def test_select_with_class(self) -> None:
        """Test that the selected fields get their descriptions from the class, if it is given."""
        fields = Fields(Field("prop"), Field("attr", description="attribute"))
        assert fields.select(["prop"])[0].description == ""
        assert [field.description for field in fields.select(["prop", "attr"], cls=Documented)] == \
            ["$p$, a property", "attribute"]

    def test_all_fields_described(self) -> None:
        """Every field of the PTtools classes should have a description, either explicit or from a docstring."""
        for module in ("pttools.bubble", "pttools.models", "pttools.omgw0", "pttools.ssm"):
            importlib.import_module(module)
        stack: list[type] = [Extractable]
        while stack:
            cls = stack.pop()
            stack.extend(cls.__subclasses__())
            for field in cls.FIELDS.select(list(cls.FIELDS), cls=cls):
                with self.subTest(field=f"{cls.__name__}.{field.name}"):
                    assert field.description


class FieldsTest(unittest.TestCase):
    """Tests for the selection of fields."""

    fields: Fields

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.fields = Fields(
            Field("a", presets={Preset.MINIMAL, Preset.FULL}),
            Field("b", presets={Preset.FULL}),
            Field("c", presets={Preset.INIT}),
            Field("d"),
        )

    def test_override_keeps_position(self) -> None:
        """Test that overriding a field keeps its original position."""
        fields = Fields(self.fields, Field("b", presets={Preset.MINIMAL}))
        assert list(fields) == ["a", "b", "c", "d"]
        assert [field.name for field in fields.preset(Preset.MINIMAL)] == ["a", "b"]

    def test_select_custom_field(self) -> None:
        """Test that custom fields can be selected alongside the presets."""
        custom = Field("custom", getter="real")
        selected = self.fields.select([Preset.MINIMAL, custom])
        assert [field.name for field in selected] == ["a", "custom"]
        assert custom.get(1.5) == 1.5

    def test_select_names_and_presets(self) -> None:
        """Test that names and presets can be mixed, and the fields are not duplicated."""
        selected = self.fields.select(["d", Preset.FULL, "a"])
        assert [field.name for field in selected] == ["d", "a", "b"]

    def test_select_preset(self) -> None:
        """Test that selecting a preset gives its fields."""
        assert [field.name for field in self.fields.select(Preset.FULL)] == ["a", "b"]

    def test_select_preset_as_str(self) -> None:
        """Test that a preset can be selected by its name as a string."""
        assert [field.name for field in self.fields.select("init")] == ["c"]

    def test_select_unknown(self) -> None:
        """Test that selecting an unknown field raises an error."""
        with pytest.raises(KeyError):
            self.fields.select(["a", "unknown"])

    def test_call(self) -> None:
        """Test that a field can call its getter method."""
        field = Field("x", getter="conjugate", call=True)
        assert field.get(1 + 2j) == 1 - 2j

    def test_call_with_function(self) -> None:
        """Test that calling is rejected for a function getter."""
        with pytest.raises(ValueError, match="call=True is valid only for attributes"):
            Field("x", getter=abs, call=True)

    def test_index(self) -> None:
        """Test that a field can take an element of a tuple returned by its getter."""
        assert Field("x", getter="as_integer_ratio", call=True, index=1).get(0.75) == 4
        assert Field("x", getter=divmod_by_3, index=0).get(7) == 2

    def test_invalid_array_type(self) -> None:
        """Test that an array of strings is rejected."""
        with pytest.raises(ValueError, match="must be numerical"):
            Field("x", type=FieldType.STR, shape=FieldShape.ARRAY)

    def test_ragged_without_axis(self) -> None:
        """Test that a ragged field without an axis is rejected."""
        with pytest.raises(ValueError, match="must have an axis"):
            Field("x", shape=FieldShape.RAGGED)


class FieldDefinitionsTest(unittest.TestCase):
    """Tests for the field definitions of the PTtools classes."""

    def test_spectrum_minimal(self) -> None:
        """Test the minimal fields of the spectra."""
        names = [field.name for field in SPECTRUM_FIELDS.preset(Preset.MINIMAL)]
        assert set(names) == {*SPECTRUM_MINIMAL_PARAMS, "y", "omgw0_h2"}

    def test_spectrum_fields_f(self) -> None:
        """Test the fields of the spectra that have been given the frequencies."""
        names = [field.name for field in SPECTRUM_FIELDS_F.preset(Preset.MINIMAL)]
        assert set(names) == {*SPECTRUM_MINIMAL_PARAMS, "f", "omgw0_h2"}
        init_names = [field.name for field in SPECTRUM_FIELDS_F.preset(Preset.INIT)]
        assert "f" in init_names
        assert "y" not in init_names
        assert SPECTRUM_FIELDS_F["f"].shape == FieldShape.GRID
        assert SPECTRUM_FIELDS_F["y"].shape == FieldShape.ARRAY
        assert SPECTRUM_FIELDS_F["y"].presets == {Preset.FULL}
        assert SPECTRUM_FIELDS_F["omgw0_h2"].axis == "f"
        # The original fields are not affected.
        assert SPECTRUM_FIELDS_Y["y"].shape == FieldShape.GRID
        assert SPECTRUM_FIELDS_Y["omgw0_h2"].axis == "y"

    def test_spectrum_fields_y_alias(self) -> None:
        """Test that the fields of the spectra with shared y are available under both names."""
        assert SPECTRUM_FIELDS_Y is SPECTRUM_FIELDS
        assert Spectrum.FIELDS_Y is Spectrum.FIELDS
        assert Spectrum.FIELDS_F is SPECTRUM_FIELDS_F

    def test_ssm_spectrum_minimal(self) -> None:
        """Test the minimal fields of the SSM spectra."""
        names = [field.name for field in SSM_SPECTRUM_FIELDS.preset(Preset.MINIMAL)]
        assert set(names) == {"v_wall", "alpha_n", "beta_tilde", "r_star", "cs2", "css2_Tn", "csb2_Tn", "y", "pow_gw"}

    def test_bubble_profiles(self) -> None:
        """Test that the bubble profiles are minimal ragged fields."""
        for name in ("v", "w", "xi"):
            field = BUBBLE_FIELDS[name]
            assert field.shape == FieldShape.RAGGED
            assert Preset.MINIMAL in field.presets


class ModelFieldsTest(unittest.TestCase):
    """Tests for the extraction of the model fields."""

    def test_bag_export_keys(self) -> None:
        """The JSON export should contain the same keys as before the field definitions."""
        model = BagModel(a_s=1.1, a_b=1, V_s=1)
        assert list(model.export()) == [
            "name", "label_latex", "label_unicode", "datetime", "T_min", "T_max",
            "restrict_to_valid", "silence_temp", "temperature_is_physical", "temperature_unit_gev",
            "T_ref", "T_crit", "V_s", "V_b", "w_crit", "w_min", "w_max", "w_min_s", "w_min_b",
            "w_max_s", "w_max_b", "alpha_n_min", "w_at_alpha_n_min", "a_s", "a_b"
        ]

    def test_const_cs_init(self) -> None:
        """Recreating a model from the INIT fields should give the same model."""
        model = ConstCSModel(css2=1/3, csb2=1/4, a_s=1.5, a_b=1, V_s=1, alpha_n_min=0.01)
        init = model.extract(Preset.INIT)
        assert init["css2"] == 1 / 3
        assert init["csb2"] == 1 / 4
        init.pop("restrict_to_valid")
        init.pop("silence_temp")
        init.pop("temperature_is_physical")
        init.pop("temperature_unit_gev")
        model2 = ConstCSModel(**init)
        assert model2.export(fields=Preset.INIT) == model.export(fields=Preset.INIT)
        assert model2.alpha_n_min == model.alpha_n_min


class BubbleFieldsTest(unittest.TestCase):
    """Tests for the extraction of the bubble fields."""

    def test_datetime_has_time_zone(self) -> None:
        """Test that the exported datetimes of the bubble and the model have a time zone."""
        bubble = Bubble(BagModel(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1)
        data = bubble.export(fields=["datetime"], model_fields=["datetime"])
        assert data["datetime"].tzinfo is not None
        assert data["model"]["datetime"].tzinfo is not None

    def test_extract_minimal(self) -> None:
        """Test that the minimal fields of a bubble can be extracted."""
        bubble = Bubble(BagModel(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1)
        data = bubble.extract()
        assert data["v_wall"] == 0.5
        assert data["alpha_n"] == 0.1
        np.testing.assert_array_equal(data["xi"], bubble.xi)

    def test_export_json_nested_model(self) -> None:
        """Test that the JSON export of a bubble contains the model as a nested dictionary."""
        bubble = Bubble(BagModel(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1)
        data = bubble.export(fields=Preset.MINIMAL, model_fields="name")
        assert data["model"] == {"name": "bag"}
        assert "v" in data


if __name__ == "__main__":
    unittest.main()
