"""Setup Sphinx."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
import typing as tp
import warnings

from matplotlib.sphinxext.plot_directive import PlotDirective
from sphinx_gallery._doctree_links import code_links_block, depart_code_links_block_html, visit_code_links_block_html
from sphinx_gallery.directives import depart_imgsg_html, imgsgnode, visit_imgsg_html

import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from sphinx.application import Sphinx
    from sphinx.config import Config

from pttools.docs.backreferences import patch_sphinx_gallery
from pttools.docs.fields import add_field_docs, note_field_dependencies
from pttools.docs.lock import setup_source_lock
from pttools.docs.minigallery import add_minigalleries, remove_duplicate_minigalleries
from pttools.docs.paths import LINT_DIR_NAME, NO_PLOTS_ENV_VAR, SPHINX_LOG_ENV_VAR, default_log_dir
from pttools.logging import setup_logging


class NoPlotDirective(PlotDirective):
    """A replacement for the Matplotlib ``plot`` directive, which does not generate the figure.

    The arguments and options are validated as by the original directive, but the figure script is not run,
    and nothing is added to the document.
    Running the figure scripts can take minutes, as they compute e.g. bubbles and spectra,
    and the figures are not needed when checking the documentation for errors with :py:mod:`pttools.docs.lint`.
    This is enabled by setting the environment variable :py:data:`pttools.docs.paths.NO_PLOTS_ENV_VAR` to ``1``.
    """

    def run(self) -> list:
        """Return no nodes, so that nothing is added to the document."""
        return []


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


def no_plots() -> bool:
    """Whether the generation of the figures of the ``plot`` directive has been disabled.

    This is set with the environment variable :py:data:`pttools.docs.paths.NO_PLOTS_ENV_VAR`.
    """
    return bool(int(os.environ.get(NO_PLOTS_ENV_VAR, "0")))


def setup_example_logging(gallery_conf: dict[str, tp.Any], fname: str | None) -> None:
    """Ensure that logging is configured before running an example.

    This is a resetter for the `reset_modules` option of Sphinx-Gallery.
    Sphinx-Gallery runs the examples as scripts without importing the `examples` package,
    which would configure logging.
    With the `parallel` option, the examples are run in worker processes,
    which do not inherit the logging configuration of the main process.
    Configuring the logging here ensures that the log messages of all examples have the same format.
    """
    setup_logging()


def setup_sphinx_logging(log_path: str | os.PathLike[str] | None = None, level: int = logging.INFO) -> Path:
    """Save the output of Sphinx to a log file.

    Sphinx has its own logging setup, which prints only messages of level INFO and above to the console,
    and which is not configured by :py:func:`pttools.logging.setup_logging`.
    This attaches a file handler to the ``sphinx`` logger, so that its messages
    are saved to ``logs/sphinx_TIMESTAMP.log`` alongside the ``docs`` directory.
    This should be called from ``docs/conf.py``.

    :param log_path: path of the log file. If None, the path is read from the environment variable
        :py:data:`SPHINX_LOG_ENV_VAR`, and if that is not set, a timestamped path is generated
        in the directory given by :py:func:`pttools.docs.paths.default_log_dir`.
    :param level: the minimum level of the messages saved to the file.
        With :py:data:`logging.DEBUG`, the file contains also the debug messages, which are not printed to the console.
    :return: path of the log file
    """
    if log_path is None:
        log_path = os.environ.get(SPHINX_LOG_ENV_VAR)
    path = Path(log_path).resolve() if log_path \
        else default_log_dir() / f"sphinx_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    sphinx_logger = logging.getLogger("sphinx")
    # The handler is added only once, even if conf.py is executed multiple times in the same process,
    # e.g. by the make mode of sphinx-build.
    if any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == path
           for handler in sphinx_logger.handlers):
        return path
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    # The Sphinx log records already contain the level prefix (e.g. "WARNING: ") and the location.
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
    sphinx_logger.addHandler(handler)
    sphinx_logger.info("Saving the Sphinx output to %s", path)
    return path


def exclude_lint_dir(app: Sphinx, config: Config) -> None:
    """Exclude the build directories of :py:mod:`pttools.docs.lint` from the documentation sources."""
    if LINT_DIR_NAME not in config.exclude_patterns:
        config.exclude_patterns.append(LINT_DIR_NAME)


def load_fluid_reference(app: Sphinx) -> None:
    """Load the reference data of the fluid solver before Sphinx-Gallery runs the examples in parallel.

    The data is generated when it's first needed, if it does not exist yet.
    If the examples are run in parallel, they would otherwise generate it simultaneously.
    This should be connected to the ``builder-inited`` event with a priority
    between those of the Sphinx-Gallery handlers ``update_gallery_conf_builder_inited`` (10),
    which applies e.g. ``-D plot_gallery=0`` to the configuration, and ``generate_gallery_rst`` (500),
    which runs the examples.

    :param app: the Sphinx application
    """
    gallery_conf = app.config.sphinx_gallery_conf
    if gallery_conf["plot_gallery"] and gallery_conf["parallel"]:
        from pttools.bubble.fluid_reference import ref  # noqa: PLC0415
        ref()


def setup_sphinx(app: Sphinx) -> None:
    """Set up the customisations of the PTtools documentation.

    To use this function, set `setup = setup_sphinx` in your `docs/conf.py`.
    """
    # Simultaneous builds wait for each other while generating files in the source directory.
    setup_source_lock(app)
    app.connect("config-inited", exclude_lint_dir)
    app.connect("autodoc-process-docstring", add_minigalleries)
    # The descriptions of the exportable fields are used as the documentation of the corresponding attributes.
    app.connect("builder-inited", add_field_docs)
    app.connect("autodoc-process-docstring", note_field_dependencies)
    app.connect("object-description-transform", remove_duplicate_minigalleries)
    app.connect("builder-inited", load_fluid_reference, priority=400)
    # The setup function of conf.py is run after the extensions have been loaded,
    # and therefore overrides the directive of matplotlib.sphinxext.plot_directive.
    if no_plots():
        app.add_directive("plot", NoPlotDirective, override=True)

    # Sphinx-Gallery registers the visitors of its image nodes only for the HTML and LaTeX builders,
    # and those of its linked code blocks only for the HTML builder,
    # and a builder falls back to the handlers of its output format only if it has no handlers of its own.
    # As sphinxcontrib-video registers handlers for the EPUB builder, the fallback is not used,
    # and the EPUB builder would not know how to render the gallery images and the code blocks of the examples.
    # The EPUB builder is based on the HTML builder, and can therefore use the HTML visitors.
    app.add_node(imgsgnode, override=True, epub=(visit_imgsg_html, depart_imgsg_html))
    app.add_node(
        code_links_block, override=True, epub=(visit_code_links_block_html, depart_code_links_block_html))
