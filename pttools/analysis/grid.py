"""Parameter grids of models, bubbles and spectra for computing datasets, see :py:mod:`pttools.analysis.dataset`."""

from collections.abc import Iterator
import dataclasses
import itertools
import math

import numpy as np

from pttools.export.records import Table
from pttools.ssm.nucleation import NucType
import pttools.type_hints as th

__all__ = ["Grid", "SpectrumParams"]


@dataclasses.dataclass(frozen=True, slots=True)
class SpectrumParams:
    """Parameters of a spectrum in addition to those of its bubble."""

    nuc_type: NucType
    beta_tilde: float
    T_star: float
    g_star: float


@dataclasses.dataclass(frozen=True, slots=True)
class Grid:
    r"""A parameter grid of models, bubbles and spectra.

    The grid is the Cartesian product of the parameter values.
    Each model has a bubble for each combination of $\alpha_n$ and $v_\text{wall}$,
    and each bubble has a spectrum for each combination of the nucleation type, $\tilde{\beta}$, $T_*$ and $g_*$.

    :param model_axes: the values of the parameters of the models by their names,
        e.g. ``{"css2": ..., "csb2": ...}`` for :py:class:`~pttools.models.const_cs.ConstCSModel`
    :param alpha_ns: $\alpha_n$, transition strengths
    :param v_walls: $v_\text{wall}$, wall speeds
    :param beta_tildes: $\tilde{\beta}$, nucleation rate parameters
    :param T_stars: $T_*$, temperatures at the time of GW production in GeV
    :param g_stars: $g_*$, degrees of freedom at the time of GW production
    :param nuc_types: nucleation types
    """

    model_axes: dict[str, th.FloatArr1D]
    alpha_ns: th.FloatArr1D
    v_walls: th.FloatArr1D
    beta_tildes: th.FloatArr1D
    T_stars: th.FloatArr1D
    g_stars: th.FloatArr1D
    nuc_types: tuple[NucType, ...] = tuple(NucType)

    def __post_init__(self) -> None:
        """Validate that each parameter has at least one value."""
        axes = {
            **self.model_axes,
            "alpha_ns": self.alpha_ns,
            "v_walls": self.v_walls,
            "beta_tildes": self.beta_tildes,
            "T_stars": self.T_stars,
            "g_stars": self.g_stars,
            "nuc_types": np.asarray(self.nuc_types),
        }
        empty = [name for name, values in axes.items() if np.size(values) == 0]
        if empty:
            raise ValueError(f"Each parameter of the grid must have at least one value. Empty: {empty}")

    @property
    def n_models(self) -> int:
        """Number of models."""
        return math.prod(np.size(values) for values in self.model_axes.values())

    @property
    def n_bubbles(self) -> int:
        """Number of bubbles."""
        return self.n_models * self.alpha_ns.size * self.v_walls.size

    @property
    def n_spectra_per_bubble(self) -> int:
        """Number of spectra of each bubble."""
        return len(self.nuc_types) * self.beta_tildes.size * self.T_stars.size * self.g_stars.size

    @property
    def n_spectra(self) -> int:
        """Number of spectra."""
        return self.n_bubbles * self.n_spectra_per_bubble

    def model_params(self) -> Iterator[dict[str, float]]:
        """The parameters of the models, with the last parameter of :py:attr:`model_axes` varying fastest."""
        names = list(self.model_axes)
        for values in itertools.product(*(np.asarray(values).tolist() for values in self.model_axes.values())):
            yield dict(zip(names, values, strict=True))

    def bubble_params(self) -> Iterator[tuple[float, float]]:
        r"""The parameters $(\alpha_n, v_{\text{wall}})$ of the bubbles of each model."""
        return itertools.product(self.alpha_ns.tolist(), self.v_walls.tolist())

    def spectrum_params(self) -> list[SpectrumParams]:
        """The parameters of the spectra of each bubble."""
        return [
            SpectrumParams(nuc_type=nuc_type, beta_tilde=beta_tilde, T_star=T_star, g_star=g_star)
            for nuc_type, beta_tilde, T_star, g_star in itertools.product(
                self.nuc_types, self.beta_tildes.tolist(), self.T_stars.tolist(), self.g_stars.tolist())
        ]

    def n_rows(self) -> dict[str, int]:
        """Number of rows of each table of the file."""
        return {Table.MODELS: self.n_models, Table.BUBBLES: self.n_bubbles, Table.SPECTRA_F: self.n_spectra}
