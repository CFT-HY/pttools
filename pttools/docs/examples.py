"""Utilities for the examples of the documentation."""

import sys

__all__ = ["is_sphinx_gallery"]


def is_sphinx_gallery() -> bool:
    """Whether the code is being run as a Sphinx-Gallery example for building the documentation.

    Sphinx-Gallery executes each example in a fake ``__main__`` module, which does not have ``__file__``,
    and replaces ``sys.modules["__main__"]`` with it during the execution.
    This can be used for skipping the computations of slow examples,
    so that their code is still shown in the documentation.

    :return: True if Sphinx-Gallery is executing an example
    """
    return "sphinx_gallery" in sys.modules and not hasattr(sys.modules.get("__main__"), "__file__")
