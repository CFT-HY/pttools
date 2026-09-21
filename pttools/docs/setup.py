"""Setup Sphinx."""

import logging
import os
import time
import typing as tp
import warnings

from sphinx_gallery.directives import depart_imgsg_html, imgsgnode, visit_imgsg_html

if tp.TYPE_CHECKING:
    from sphinx.application import Sphinx

from pttools.docs.backreferences import patch_sphinx_gallery
from pttools.docs.minigallery import add_minigalleries, remove_duplicate_minigalleries
from pttools.logging import setup_logging


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


#: Environment variable with which the path of the Sphinx log file can be set,
#: e.g. by :py:mod:`pttools.docs.lint`.
SPHINX_LOG_ENV_VAR: str = "PTTOOLS_SPHINX_LOG"


def setup_sphinx_logging(log_path: str | None = None, level: int = logging.INFO) -> str:
    """Save the output of Sphinx to a log file.

    Sphinx has its own logging setup, which prints only messages of level INFO and above to the console,
    and which is not configured by :py:func:`pttools.logging.setup_logging`.
    This attaches a file handler to the ``sphinx`` logger, so that its messages
    are saved to ``logs/sphinx_TIMESTAMP.log`` in the repository.
    This should be called from ``docs/conf.py``.

    :param log_path: path of the log file. If None, the path is read from the environment variable
        :py:data:`SPHINX_LOG_ENV_VAR`, and if that is not set, a timestamped path is generated.
    :param level: the minimum level of the messages saved to the file.
        With :py:data:`logging.DEBUG`, the file contains also the debug messages, which are not printed to the console.
    :return: path of the log file
    """
    if log_path is None:
        log_path = os.environ.get(SPHINX_LOG_ENV_VAR)
    if not log_path:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
        log_path = os.path.join(log_dir, f"sphinx_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    sphinx_logger = logging.getLogger("sphinx")
    # The handler is added only once, even if conf.py is executed multiple times in the same process,
    # e.g. by the make mode of sphinx-build.
    if any(isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_path)
           for handler in sphinx_logger.handlers):
        return log_path
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    # The Sphinx log records already contain the level prefix (e.g. "WARNING: ") and the location.
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
    sphinx_logger.addHandler(handler)
    sphinx_logger.info("Saving the Sphinx output to %s", log_path)
    return log_path


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
