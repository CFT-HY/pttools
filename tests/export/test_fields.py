"""Tests for the field definitions and the extraction of the fields."""

import unittest

import numpy as np

from pttools.bubble import BUBBLE_FIELDS, Bubble
from pttools.models import BagModel, ConstCSModel
from pttools.omgw0 import SPECTRUM_FIELDS
from pttools.ssm import SSM_SPECTRUM_FIELDS
from pttools.utils.fields import Field, Fields, FieldShape, FieldType, Preset

#: The minimal parameters of the spectra
SPECTRUM_MINIMAL_PARAMS: tuple[str, ...] = (
    "v_wall", "alpha_n", "beta_tilde", "r_star", "T_star", "g_star", "cs2", "css2_Tn", "csb2_Tn")


class FieldsTest(unittest.TestCase):
    """Tests for the selection of fields."""

    fields: Fields

    @classmethod
    def setUpClass(cls) -> None:
        cls.fields = Fields(
            Field("a", presets={Preset.MINIMAL, Preset.FULL}),
            Field("b", presets={Preset.FULL}),
            Field("c", presets={Preset.INIT}),
            Field("d"),
        )

    def test_override_keeps_position(self) -> None:
        fields = Fields(self.fields, Field("b", presets={Preset.MINIMAL}))
        self.assertEqual(list(fields), ["a", "b", "c", "d"])
        self.assertEqual([field.name for field in fields.preset(Preset.MINIMAL)], ["a", "b"])

    def test_select_custom_field(self) -> None:
        custom = Field("custom", getter="real")
        selected = self.fields.select([Preset.MINIMAL, custom])
        self.assertEqual([field.name for field in selected], ["a", "custom"])
        self.assertEqual(custom.get(1.5), 1.5)

    def test_select_names_and_presets(self) -> None:
        selected = self.fields.select(["d", Preset.FULL, "a"])
        self.assertEqual([field.name for field in selected], ["d", "a", "b"])

    def test_select_preset(self) -> None:
        self.assertEqual([field.name for field in self.fields.select(Preset.FULL)], ["a", "b"])

    def test_select_preset_as_str(self) -> None:
        self.assertEqual([field.name for field in self.fields.select("init")], ["c"])

    def test_select_unknown(self) -> None:
        with self.assertRaises(KeyError):
            self.fields.select(["a", "unknown"])

    def test_invalid_array_type(self) -> None:
        with self.assertRaises(ValueError):
            Field("x", type=FieldType.STR, shape=FieldShape.ARRAY)

    def test_ragged_without_axis(self) -> None:
        with self.assertRaises(ValueError):
            Field("x", shape=FieldShape.RAGGED)


class FieldDefinitionsTest(unittest.TestCase):
    """Tests for the field definitions of the PTtools classes."""

    def test_spectrum_minimal(self) -> None:
        names = [field.name for field in SPECTRUM_FIELDS.preset(Preset.MINIMAL)]
        self.assertEqual(set(names), {*SPECTRUM_MINIMAL_PARAMS, "y", "omgw0_h2"})

    def test_ssm_spectrum_minimal(self) -> None:
        names = [field.name for field in SSM_SPECTRUM_FIELDS.preset(Preset.MINIMAL)]
        self.assertEqual(
            set(names),
            {"v_wall", "alpha_n", "beta_tilde", "r_star", "cs2", "css2_Tn", "csb2_Tn", "y", "pow_gw"}
        )

    def test_bubble_profiles(self) -> None:
        for name in ("v", "w", "xi"):
            field = BUBBLE_FIELDS[name]
            self.assertEqual(field.shape, FieldShape.RAGGED)
            self.assertIn(Preset.MINIMAL, field.presets)


class ModelFieldsTest(unittest.TestCase):
    """Tests for the extraction of the model fields."""

    def test_bag_export_keys(self) -> None:
        """The JSON export should contain the same keys as before the field definitions."""
        model = BagModel(a_s=1.1, a_b=1, V_s=1)
        self.assertEqual(
            list(model.export()),
            [
                "name", "label_latex", "label_unicode", "datetime", "T_min", "T_max",
                "restrict_to_valid", "silence_temp", "temperature_is_physical",
                "T_ref", "T_crit", "V_s", "V_b", "w_crit", "w_min", "w_max", "w_min_s", "w_min_b",
                "w_max_s", "w_max_b", "alpha_n_min", "w_at_alpha_n_min", "a_s", "a_b"
            ]
        )

    def test_const_cs_init(self) -> None:
        """Recreating a model from the INIT fields should give the same model."""
        model = ConstCSModel(css2=1/3, csb2=1/4, a_s=1.5, a_b=1, V_s=1, alpha_n_min=0.01)
        init = model.extract(Preset.INIT)
        self.assertEqual(init["css2"], 1/3)
        self.assertEqual(init["csb2"], 1/4)
        init.pop("restrict_to_valid")
        init.pop("silence_temp")
        init.pop("temperature_is_physical")
        model2 = ConstCSModel(**init)
        self.assertEqual(model2.export(fields=Preset.INIT), model.export(fields=Preset.INIT))
        self.assertEqual(model2.alpha_n_min, model.alpha_n_min)


class BubbleFieldsTest(unittest.TestCase):
    """Tests for the extraction of the bubble fields."""

    def test_cs2_Tn_bag(self) -> None:
        r"""In the bag model $c_s^2 = 1/3$ in both phases."""
        bubble = Bubble(BagModel(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1)
        data = bubble.extract()
        self.assertAlmostEqual(data["css2_Tn"], 1/3, places=15)
        self.assertAlmostEqual(data["csb2_Tn"], 1/3, places=15)

    def test_cs2_Tn_const_cs(self) -> None:
        r"""In the constant sound speed model $c_s^2$ is constant in each phase."""
        model = ConstCSModel(css2=1/3 - 0.01, csb2=1/3 - 0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        data = bubble.extract()
        self.assertAlmostEqual(data["css2_Tn"], 1/3 - 0.01, places=12)
        self.assertAlmostEqual(data["csb2_Tn"], 1/3 - 0.011, places=12)

    def test_extract_minimal(self) -> None:
        bubble = Bubble(BagModel(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1)
        data = bubble.extract()
        self.assertEqual(data["v_wall"], 0.5)
        self.assertEqual(data["alpha_n"], 0.1)
        np.testing.assert_array_equal(data["xi"], bubble.xi)

    def test_export_json_nested_model(self) -> None:
        bubble = Bubble(BagModel(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1)
        data = bubble.export(fields=Preset.MINIMAL, model_fields="name")
        self.assertEqual(data["model"], {"name": "bag"})
        self.assertIn("v", data)


if __name__ == "__main__":
    unittest.main()
