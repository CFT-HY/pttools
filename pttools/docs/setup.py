"""Setup Sphinx."""

import typing as tp
import warnings

from sphinx_gallery.directives import depart_imgsg_html, imgsgnode, visit_imgsg_html

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

    # Sphinx-Gallery registers the visitors of its image nodes only for the HTML and LaTeX builders,
    # and a builder falls back to the handlers of its output format only if it has no handlers of its own.
    # As sphinxcontrib-video registers handlers for the EPUB builder, the fallback is not used,
    # and the EPUB builder would not know how to render the gallery images.
    # The EPUB builder is based on the HTML builder, and can therefore use the HTML visitors.
    app.add_node(imgsgnode, override=True, epub=(visit_imgsg_html, depart_imgsg_html))
