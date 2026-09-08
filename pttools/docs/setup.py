"""Setup Sphinx."""

import typing as tp
import warnings

if tp.TYPE_CHECKING:
    from sphinx.application import Sphinx

from pttools.docs.backreferences import patch_sphinx_gallery
from pttools.docs.minigallery import add_minigalleries, remove_duplicate_minigalleries


def filter_warnings() -> None:
    """Remove unnecessary warnings from generated documentation.

    Remove Matplotlib agg warnings from generated documentation when using :py:func:`matplotlib.pyplot.show`.
    From: https://github.com/sphinx-gallery/sphinx-gallery/blob/master/doc/conf.py

    Also remove "invalid value encountered in multiply" warnings.
    """
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure."
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in multiply"
    )


def pre_setup(doc_modules: tp.Container[str]) -> None:
    """This should be called from `docs/conf.py`."""
    patch_sphinx_gallery(doc_modules=doc_modules)
    filter_warnings()


def setup_sphinx(app: "Sphinx") -> None:
    """Set up the customisations of the PTtools documentation.

    To use this function, set `setup = setup_sphinx` in your `docs/conf.py`.
    """
    app.connect("autodoc-process-docstring", add_minigalleries)
    app.connect("object-description-transform", remove_duplicate_minigalleries)
