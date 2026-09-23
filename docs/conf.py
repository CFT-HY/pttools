"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# The imports have to be after the sys.path manipulation below.
# ruff: noqa: E402

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

from datetime import date
import logging
import os.path
import sys
import tomllib
import typing as tp

from matplotlib.animation import FFMpegWriter

# import plotly.io as pio
from sphinx_gallery.sorting import ExplicitOrder

DOCS_DIR: str = os.path.dirname(os.path.abspath(__file__))
REPO_DIR: str = os.path.dirname(DOCS_DIR)
EXAMPLES_DIR: str = os.path.join(REPO_DIR, "examples")
TESTS_DIR: str = os.path.join(REPO_DIR, "tests")
sys.path.insert(0, REPO_DIR)

from pttools.docs.intersphinx import INTERSPHINX_MAPPING, IntersphinxMapping
from pttools.docs.links import EXTLINKS, LINKCHECK_ALLOWED_REDIRECTS, ExtLinks
from pttools.docs.setup import pre_setup, setup_sphinx, setup_sphinx_logging
from pttools.logging import setup_logging
from pttools.utils.system import IS_GITHUB_ACTIONS, PTTOOLS_DIR

setup_logging()
logger: logging.Logger = logging.getLogger(__name__)
# The Sphinx output is saved to logs/sphinx_TIMESTAMP.log. See also pttools/docs/lint.py.
setup_sphinx_logging()

#: The packages of this repository, which are documented in this documentation.
#: Sphinx-Gallery creates hyperlinks from the examples and mini-galleries for the objects of these packages.
DOC_MODULES: tuple[str, ...] = ("docs", "examples", "pttools", "tests")
pre_setup(doc_modules=DOC_MODULES)

# Create a directory for static files to avoid a warning when building.
os.makedirs(os.path.join(DOCS_DIR, "_static"), exist_ok=True)

# -- Project information -----------------------------------------------------

project = "PTtools"
file: tp.IO[tp.Any]
with open(os.path.join(REPO_DIR, "AUTHORS")) as file:
    _authors = file.read().splitlines()
author = f"{', '.join(_authors[:-1])} & {_authors[-1]}"
copyright = f"2015-{date.today().year}, {author}"
with open (os.path.join(REPO_DIR, "pyproject.toml"), "rb") as file:
    version = tomllib.load(file)["project"]["version"]
release = version


# -- General configuration ---------------------------------------------------

setup = setup_sphinx

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    # Automatic documentation for Python code
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    # "sphinx_autodoc_typehints",
    # Automatic labeling for documentation sections
    "sphinx.ext.autosectionlabel",
    # External links
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    # Mathematics rendering
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinx_math_dollar",
    # Markdown support can be enabled by uncommenting the line below.
    # https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html#using-markdown-with-sphinx
    # "myst_parser"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    # Apidoc generates a table of contents file for each package,
    # but the packages are included in the main toctree directly.
    'gen_modules/*/modules.rst',
]

suppress_warnings = [
    # Automatic section labeling produces duplicated labels. This silences the warnings from those.
    # https://github.com/sphinx-doc/sphinx/issues/7728
    # https://github.com/sphinx-doc/sphinx/issues/7697
    "autosectionlabel.*",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- LaTeX -------------------------------------------------------------------

# For Unicode support
latex_engine = "xelatex"


# -- Math --------------------------------------------------------------------

# This may not work unless changed to "mathjax_config", but that gives warnings with MathJax 3
mathjax3_config = {
    'tex2jax': {
        'inlineMath': [["\\(", "\\)"]],
        'displayMath': [["\\[", "\\]"]],
    },
}


# -- Apidoc  -----------------------------------------------------------------
apidoc_modules = [
    {
        "path": PTTOOLS_DIR,
        "destination": "gen_modules/pttools"
    },
    {
        # Only the utilities are documented, as the examples themselves are in the gallery,
        # and importing them for autodoc would run them a second time.
        "path": EXAMPLES_DIR,
        "destination": "gen_modules/examples",
        "exclude_patterns": [os.path.join(EXAMPLES_DIR, "*", "*")]
    },
    {
        "path": TESTS_DIR,
        "destination": "gen_modules/tests"
    },
    {
        # This file is excluded, since importing it for autodoc would run it a second time.
        # The figure scripts are excluded, as they are already included with the plot directive.
        "path": DOCS_DIR,
        "destination": "gen_modules/docs",
        "exclude_patterns": [os.path.join(DOCS_DIR, "conf.py"), os.path.join(DOCS_DIR, "fig")]
    }
]
# apidoc_max_depth = 6
apidoc_module_first = True
apidoc_separate_modules = True

# -- Autodoc -----------------------------------------------------------------

# autodoc_default_options = {
#     # This would result in duplicate class descriptions when using a template.
#     "members": True,
#     "show-inheritance": True,
#     "undoc-members": True,
# }
autoclass_content = "both"
autodoc_preserve_defaults = True
autodoc_typehints = "description"


# def skip(app, what, name, obj, would_skip, options):
#     """Enabling the documentation of __init__ functions
#     https://stackoverflow.com/a/5599712/
#     """
#     if name == "__init__":
#         return False
#     return would_skip
#
#
# def setup(app):
#     app.connect("autodoc-skip-member", skip)


# -- Type hints -----------------------------------------------------------------

# always_document_param_types = True
# always_use_bars_union = True  # This is the default on Python 3.14 ->
# typehints_defaults = "braces"


# -- Other -------------------------------------------------------------------

# Sphinx requires base URLs and caption strings to contain exactly one "%s",
# and all other "%" need to be escaped as "%%".
extlinks: ExtLinks = EXTLINKS
extlinks_detect_hardcoded_links: bool = True
intersphinx_mapping: IntersphinxMapping = INTERSPHINX_MAPPING
linkcheck_allowed_redirects: dict[str, str] = LINKCHECK_ALLOWED_REDIRECTS
# The authentication info could be set up to work on the CI build
# https://docs.github.com/en/actions/reference/authentication-in-a-workflow
# linkcheck_auth = []
linkcheck_ignore: list[str] = [
    # These websites don't allow crawlers
    # r"https://academic.oup.com/book/*",
    "https://link.aps.org/*",
    "https://www.aka.fi/*",
    "https://www.intel.com/*",
    # The private Bitbucket repos will also return 404 without authentication
    "https://bitbucket.org/cgowling/pttools_omgw0_addons/*",
    "https://bitbucket.org/hindmars/sound-shell-model/*",
    # These links redirect to sites that do not allow crawlers
    "https://doi.org/10.1086/344402",
    "https://doi.org/10.1093/acprof:oso/9780198528906.001.0001",
    # The anchors are valid but not detected by Sphinx.
    "https://github.com/scipy/scipy/blob/v1.8.0/scipy/interpolate/fitpack/*",
    r"https://scicomp\.stackexchange\.com/*",
    r"https://stackoverflow\.com/*",
]
if IS_GITHUB_ACTIONS:
    linkcheck_ignore += [
        r"https://akareport\.aka\.fi/ibi_apps/WFServlet*",
        r"https://www\.intel\.com/*",
        r"https://gtr\.ukri\.org/*",
        r"https://stfc\.ukri\.org/",
    ]

linkcheck_retries = 5
# Timeout had to be increased from 5 to prevent errors with slow ArXiv links
linkcheck_timeout = 20
linkcheck_workers = 10

# pio.renderers.default = "sphinx_gallery"
# pio.renderers.default = "sphinx_gallery_png"

# show_memory = IS_GITHUB_ACTIONS
show_memory = True

sphinx_gallery_conf = {
    "backreferences_dir": "gen_modules/backreferences",
    "compress_images": ("images", "thumbnails"),
    "doc_module": DOC_MODULES,
    "examples_dirs": EXAMPLES_DIR,
    "filename_pattern": ".*",
    "gallery_dirs": "auto_examples",
    "ignore_pattern": r"(__init__\.py|utils\.py|p_s_scan_dev\.py|droplet|standard_model|entropy|reverse)",
    # "image_scrapers": ("matplotlib", "plotly.io._sg_scraper.plotly_sg_scraper"),
    "image_srcset": ["2x"],
    # "line_numbers": True,
    "matplotlib_animations": (True, "mp4"),
    # Parallelism cannot be enabled simultaneously with "show_memory".
    # It may also produce errors with some IDEs:
    # https://stackoverflow.com/questions/31080829/python-error-io-unsupportedoperation-fileno
    "parallel": not show_memory,
    # This has to be set in order to avoid a warning when disabling it with a command line option.
    # https://sphinx-gallery.github.io/stable/configuration.html#building-without-executing-examples
    "plot_gallery": "True",
    # By default, Sphinx-Gallery refers to the objects by the shortest name with which they are accessible,
    # e.g. "pttools.models.BagModel", but Sphinx documents them by the module in which they are defined,
    # e.g. "pttools.models.bag.BagModel". Without this, the hyperlinks from the examples to the API documentation
    # cannot be resolved, and the backreferences, that the mini-galleries are based on, are stored under names
    # that don't correspond to the documented objects.
    "prefer_full_module": {rf"^{module}\." for module in DOC_MODULES},
    # The None values mean that the objects are documented in this documentation instead of an external one.
    "reference_url": dict.fromkeys(DOC_MODULES),
    # Ensure that logging is configured for all examples, so that their log messages have the same format.
    # The function is given as a string, since the configuration has to be picklable.
    "reset_modules": ("matplotlib", "seaborn", "pttools.docs.setup.setup_example_logging"),
    # "run_stale_examples": True
    "show_api_usage": True,
    "show_memory": show_memory,
    "subsection_order": ExplicitOrder([
        "../examples/basic",
        "../examples/const_cs",
        # "../examples/standard_model",
        "../examples/props",
        # "../examples/entropy",
        "../examples/solvers",
        "../examples/low_k",
        "../examples/gksvdv",
        # "../examples/reverse",
        # "*"
    ])
}
autosummary_generate = True

if not FFMpegWriter.isAvailable():
    logger.error("FFmpeg is not available. Animations will not be rendered in the documentation.")

# numpydoc_show_class_members = False
