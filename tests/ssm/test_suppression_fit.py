r"""Tests for fitting the suppression factor and for combining suppression datasets."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.ssm.suppression import (
    CORREIA_2025,
    NO_HYBRIDS_EXT,
    NO_HYBRIDS_EXT_CORREIA_2025,
    SUPPRESSIONS,
    WITH_HYBRIDS,
    WITH_HYBRIDS_CORREIA_2025,
    Suppression,
    SuppressionFitMethod,
    fit_suppression,
    lattice_artifact_start,
)


def model_spectrum(y: np.ndarray) -> np.ndarray:
    """A broken power law with a peak at $z = 10$, similar to a GW power spectrum."""
    return y**3 / (1 + (y / 10)**6)


def artifact(y: np.ndarray) -> np.ndarray:
    """A separate peak at high $z$, similar to the lattice artifact of the simulations."""
    return 1e-2 * np.exp(-0.5 * (np.log(y / 4000) / 0.2)**2)


class LatticeArtifactTest(unittest.TestCase):
    """Tests for detecting the high-frequency lattice artifact."""

    y = np.linspace(1, 6000, 4000)

    def test_detects_artifact(self) -> None:
        """A separate peak is detected, and it starts between the main peak and the artifact."""
        i = lattice_artifact_start(self.y, model_spectrum(self.y) + artifact(self.y))
        assert i is not None
        assert 300 < self.y[i] < 4000

    def test_no_artifact(self) -> None:
        """A monotonically decreasing tail is not detected as an artifact."""
        assert lattice_artifact_start(self.y, model_spectrum(self.y)) is None

    def test_noise_is_not_artifact(self) -> None:
        """Small fluctuations are not detected as an artifact."""
        rng = np.random.default_rng(1234)
        p = model_spectrum(self.y) * np.exp(0.3 * rng.standard_normal(self.y.size))
        assert lattice_artifact_start(self.y, p) is None

    def test_shape_mismatch(self) -> None:
        """Arrays of different shapes are rejected."""
        with pytest.raises(ValueError, match="same shape"):
            lattice_artifact_start(self.y, self.y[:-1])


class FitSuppressionTest(unittest.TestCase):
    """Tests for fitting the suppression factor."""

    y_ref = np.linspace(1, 6000, 4000)
    y_model = np.logspace(-1, 4, 1000)
    factor = 0.3

    def fit(self, p_ref: np.ndarray, **kwargs: tp.Any) -> float:
        """Fit the suppression factor of the reference spectrum with respect to the model spectrum."""
        return fit_suppression(self.y_ref, p_ref, self.y_model, model_spectrum(self.y_model), **kwargs).factor

    def test_exact(self) -> None:
        """A scaled copy of the model gives the scaling factor with both methods."""
        p_ref = self.factor * model_spectrum(self.y_ref)
        for method in SuppressionFitMethod:
            with self.subTest(method=method):
                assert self.fit(p_ref, method=method) == pytest.approx(self.factor, rel=1e-3)

    def test_artifact_excluded(self) -> None:
        """The lattice artifact does not affect the fit, unless it's explicitly included."""
        p_ref = self.factor * model_spectrum(self.y_ref) + artifact(self.y_ref)
        result = fit_suppression(self.y_ref, p_ref, self.y_model, model_spectrum(self.y_model))
        assert result.factor == pytest.approx(self.factor, rel=1e-3)
        assert np.isfinite(result.y_artifact)
        assert result.y_max < result.y_artifact
        assert result.rms_log10_residual < 0.01
        included = self.fit(p_ref, method=SuppressionFitMethod.LOG_LSQ, exclude_artifact=False)
        assert included > 1.1 * self.factor

    def test_range(self) -> None:
        """The fit range is respected."""
        p_ref = self.factor * model_spectrum(self.y_ref)
        result = fit_suppression(self.y_ref, p_ref, self.y_model, model_spectrum(self.y_model), y_min=5, y_max=50)
        assert 5 <= result.y_min < 6
        assert 49 < result.y_max <= 50
        assert result.factor == pytest.approx(self.factor, rel=1e-3)

    def test_negative_values_ignored(self) -> None:
        """Non-positive values of the reference spectrum are excluded from the fit."""
        p_ref = self.factor * model_spectrum(self.y_ref)
        p_ref[::7] = -1
        assert self.fit(p_ref, method=SuppressionFitMethod.LOG_LSQ) == pytest.approx(self.factor, rel=1e-3)

    def test_too_few_points(self) -> None:
        """A fit range with less than two points is rejected."""
        with pytest.raises(ValueError, match="at least two points"):
            self.fit(model_spectrum(self.y_ref), y_min=1.5, y_max=2)


class MergeTest(unittest.TestCase):
    """Tests for combining suppression datasets."""

    def test_merge_replaces(self) -> None:
        """The points of the other dataset replace the coinciding points."""
        base = Suppression(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.1, 0.2]), np.array([1., 2., 3.]), name="a")
        other = Suppression(np.array([0.2, 0.5]), np.array([0.1, 0.5]), np.array([20., 50.]), name="b")
        merged = base.merge(other, name="c")
        assert merged.name == "c"
        assert merged.v_walls.size == 4
        assert sorted(merged.suppressions.tolist()) == [1., 3., 20., 50.]

    def test_correia_datasets(self) -> None:
        """The combined datasets contain the new values at the simulated points, and the old values elsewhere."""
        for base, merged in ((NO_HYBRIDS_EXT, NO_HYBRIDS_EXT_CORREIA_2025), (WITH_HYBRIDS, WITH_HYBRIDS_CORREIA_2025)):
            with self.subTest(dataset=merged.name):
                assert merged in SUPPRESSIONS
                assert merged.v_walls.size == base.v_walls.size
                for v_wall, alpha_n, sup in zip(
                        CORREIA_2025.v_walls, CORREIA_2025.alpha_ns, CORREIA_2025.suppressions, strict=True):
                    assert merged.suppression(float(v_wall), float(alpha_n)) == pytest.approx(sup, rel=1e-12)
                assert merged.suppression(0.6, 0.1) == pytest.approx(base.suppression(0.6, 0.1), rel=1e-12)

    def test_correia_values(self) -> None:
        r"""Reference values of the suppression factors fitted to the simulations of :correia_2025:`\ `.

        These were computed on 2026-10-11 with the default settings of
        :py:mod:`pttools.ssm.suppression.suppression_ssm_data.suppression_hydro_calculator`.
        Recompute the dataset if the SSM spectra change.
        """
        assert CORREIA_2025.v_walls.tolist() == [0.92, 0.44]
        assert CORREIA_2025.alpha_ns.tolist() == [0.67, 0.5]
        np.testing.assert_allclose(CORREIA_2025.suppressions, [0.33573, 0.012208], rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
