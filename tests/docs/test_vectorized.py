"""Unit tests for the documentation of the vectorized functions."""

import inspect
import types

import pytest

pytest.importorskip("sphinx", reason="Sphinx is not installed")

from pttools.bubble import energy_budget, shock
from pttools.bubble.alpha import alpha_limits_bag
from pttools.docs.vectorized import add_vectorized_functions, vectorized_function


def test_add_vectorized_functions() -> None:
    """The vectorized functions of a module are added with autofunction, and the other members are not."""
    lines = ["Docstring of the module."]
    options = types.SimpleNamespace(members=True, exclude_members={"alpha_plus_min_hybrid"})
    add_vectorized_functions(
        None, "module", alpha_limits_bag.__name__, alpha_limits_bag, options, lines)
    assert lines == [
        "Docstring of the module.",
        "",
        ".. autofunction:: pttools.bubble.alpha.alpha_limits_bag.alpha_plus_max_detonation_bag",
    ]


def test_add_vectorized_functions_not_module() -> None:
    """Nothing is added to the docstrings of the other objects or of modules documented without members."""
    lines: list[str] = []
    add_vectorized_functions(
        None, "function", "pttools.bubble.shock.v_shock", shock.v_shock,
        types.SimpleNamespace(members=True), lines)
    add_vectorized_functions(
        None, "module", shock.__name__, shock,
        types.SimpleNamespace(members=None), lines)
    assert not lines


def test_add_vectorized_functions_signature() -> None:
    """The original function of a NumPy-vectorized function is set as its ``__wrapped__``."""
    add_vectorized_functions(
        None, "module", energy_budget.__name__, energy_budget,
        types.SimpleNamespace(members=["alpha_n_from_ubarf"]), [])
    assert energy_budget.alpha_n_from_ubarf.__wrapped__ is energy_budget.alpha_n_from_ubarf.pyfunc
    assert energy_budget.alpha_n_from_ubarf.__module__ == energy_budget.__name__


def test_vectorized_function() -> None:
    """The original functions of the NumPy- and Numba-vectorized functions are found, but not those of njit."""
    assert vectorized_function(shock.v_shock) is shock.v_shock.pyfunc
    assert vectorized_function(alpha_limits_bag.alpha_plus_max_detonation_bag) \
        is inspect.unwrap(alpha_limits_bag.alpha_plus_max_detonation_bag)
    assert vectorized_function(alpha_limits_bag.alpha_n_max_detonation_bag) is None
    assert vectorized_function(test_vectorized_function) is None
    assert vectorized_function(alpha_limits_bag) is None
