"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# pylint: disable=invalid-name, redefined-builtin

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

from datetime import date
import logging
import os.path
import sys
import tomllib
import warnings

from matplotlib.animation import FFMpegWriter
# import plotly.io as pio
from sphinx.application import Sphinx
from sphinx_gallery.sorting import ExplicitOrder

DOCS_DIR: str = os.path.dirname(os.path.abspath(__file__))
REPO_DIR: str = os.path.dirname(DOCS_DIR)
EXAMPLES_DIR: str = os.path.join(REPO_DIR, "examples")
TESTS_DIR: str = os.path.join(REPO_DIR, "tests")
sys.path.insert(0, REPO_DIR)

from docs.backreferences import patch_sphinx_gallery
from docs.utils import DOC_MODULES
from docs.links import ExtLinks, arxiv_link, convert_extlinks, doi_link, hdl_link
from docs.minigallery import add_minigalleries, remove_duplicate_minigalleries
from pttools.logging import setup_logging
from pttools.utils.system import IS_GITHUB_ACTIONS, PTTOOLS_DIR
setup_logging()
logger = logging.getLogger(__name__)
patch_sphinx_gallery()

# Create a directory for static files to avoid a warning when building.
os.makedirs(os.path.join(DOCS_DIR, "_static"), exist_ok=True)

# -- Project information -----------------------------------------------------

project = 'PTtools'
with open(os.path.join(REPO_DIR, "AUTHORS"), "r") as file:
    _authors = file.read().splitlines()
author = f"{', '.join(_authors[:-1])} & {_authors[-1]}"
copyright = f"2015-{date.today().year}, {author}"
with open (os.path.join(REPO_DIR, "pyproject.toml"), "rb") as file:
    version = tomllib.load(file)["project"]["version"]
release = version


# -- General configuration ---------------------------------------------------

def setup(app: Sphinx) -> None:
    """Set up the customisations of the PTtools documentation"""
    app.connect("autodoc-process-docstring", add_minigalleries)
    app.connect("object-description-transform", remove_duplicate_minigalleries)


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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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

# Sphinx 6.0 will require base URLs and caption strings to contain exactly one "%s",
# and all other "%" need to be escaped as "%%".
HINDMARSH_ET_AL: str = "Hindmarsh et al."
EXTLINKS_STATIC: ExtLinks = {
    # Hindmarsh articles
    "hindmarsh_2014": arxiv_link("1304.2433", HINDMARSH_ET_AL),
    "hindmarsh_2015": arxiv_link("1504.03291", HINDMARSH_ET_AL),
    "hindmarsh_2017": arxiv_link("1704.05871", HINDMARSH_ET_AL),
    "ssm": arxiv_link("1608.04735", HINDMARSH_ET_AL),
    "gw_pt_ssm": arxiv_link("1909.10040", HINDMARSH_ET_AL),
    "notes": arxiv_link("2008.09136", HINDMARSH_ET_AL),
    # Other articles
    "enqvist_1992": doi_link("10.1103/PhysRevD.45.3415", "Enqvist et al.", 1992),
    "kurki-suonio_1995": arxiv_link("hep-ph/9512202", "Kurki-Suonio & Laine", 1995),
    "maggiore_1999": arxiv_link("gr-qc/9909001", "Maggiore", 1999),
    "espinosa_2010": arxiv_link("1004.4187", "Espinosa"),
    "borsanyi_2016": arxiv_link("1606.07494", "Borsanyi et al."),
    "caprini_2016": arxiv_link("1512.06239", "Caprini et al.", 2016),
    "cornish_2017": arxiv_link("1703.09858", "Cornish & Robson"),
    "planck_2018": arxiv_link("1807.06209", "Planck 2018 results"),
    "smith_2019": arxiv_link("1908.00546", "Smith & Caldwell"),
    "caprini_2020": arxiv_link("1910.13125", "Caprini et al.", 2020),
    "giese_2020": arxiv_link("2004.06995", "Giese et al."),
    "giese_2021": arxiv_link("2010.09744", "Giese et al.", 2021),
    "gowling_2021": arxiv_link("2106.05984", "Gowling & Hindmarsh"),
    "ajmi_2022": arxiv_link("2205.04097", "Ajmi & Hindmarsh"),
    "cutting_2022": arxiv_link("2204.03396", "Cutting, Vilhonen & Weir"),
    "ai_2023": arxiv_link("2303.10171", "Ai et al."),
    "gowling_2023": arxiv_link("2209.13551", "Gowling et al.", 2023),
    "lewicki_2023": arxiv_link("2305.04924", "Lewicki et al."),
    "barni_2024": arxiv_link("2406.01596", "Barni et al."),
    "croon_2024": arxiv_link("2410.21509", "Croon & Weir"),
    "giombi_2024_cs": arxiv_link("2409.01426", "Giombi et al."),
    "giombi_2024_gr": arxiv_link("2307.12080", "Giombi & Hindmarsh", 2024),
    "barni_2026": arxiv_link("2510.21439", "Barni et al.", 2026),
    "bhusal_2026": arxiv_link("2603.22397", "Bhusal et al."),
    "correia_2026": arxiv_link("2505.17824", "Correia et al.", 2026),
    "giombi_2026": arxiv_link("2504.08037", "Giombi et al.", 2026),
    # Theses
    "gowling_phd": hdl_link("10779/uos.23309135.v1", "Gowling", 2023),
    "hakkinen_msc": hdl_link("10138/576963", "Häkkinen", 2024),
    "maki_msc": arxiv_link("2511.20436", "Mäki", 2025),
    # Other
    "lisa_conventions": ("https://gitlab.esa.int/lisa-sgs/sandbox/conventions-document", "LISA DDPC Conventions document"),
    "lisa_sci_req": ("https://www.cosmos.esa.int/web/lisa/documents", "LISA Science Requirements Document"),
    "rel_hydro_book": doi_link("10.1093/acprof:oso/9780198528906.001.0001", "Relativistic hydrodynamics: Rezzolla, Zanotti", 2013)
}
extlinks: ExtLinks = {
    **convert_extlinks(EXTLINKS_STATIC),
    # Other
    "aof_grant": (
        "https://akareport.aka.fi/ibi_apps/WFServlet?IBIF_ex=x_hakkuvaus2&CLICKED_ON=&UILANG=en&TULOSTE=HTML&HAKNRO1=%s",
        "Academy of Finland grant %s"
    ),
    "issue": ("https://github.com/CFT-HY/pttools/issues/%s", "issue %s"),
    "ssm_repo": ("https://bitbucket.org/hindmars/sound-shell-model/src/master/%s", "sound-shell-model/%s")
}
extlinks_detect_hardcoded_links: bool = True
intersphinx_mapping: dict[str, tuple[str, str | None]] = {
    "cobaya": ("https://cobaya.readthedocs.io/en/latest/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "pyinstrument": ("https://pyinstrument.readthedocs.io/en/latest/", None),
    "pylint": ("https://pylint.readthedocs.io/en/stable/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    # "yappi": ("https://yappi.readthedocs.io/en/latest/", None),
}
linkcheck_allowed_redirects: dict[str, str] = {
    "https://akareport.aka.fi/*": "https://tiedejatutkimus.fi/*",
    "https://bitbucket.org/*": "https://id.atlassian.com/*",
    "https://gitlab.esa.int/*": "https://gitlab.esa.int/users/sign_in",
    "https://www.helsinki.fi/": "https://www.helsinki.fi/en",
    "https://hdl.handle.net/*": "(https://helda.helsinki.fi/handle/*|https://sussex.figshare.com/*)",
    "https://www.ptplot.org": "https://www.ptplot.org/ptplot/",
    r"https://.*\.stackexchange.com/a/.*": r"https://.*\.stackexchange.com/questions/.*",
    "https://stackoverflow.com/a/*": "https://stackoverflow.com/questions/*",
}
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
    # This link redirects to a site that does not allow crawlers
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
    "reference_url": {module: None for module in DOC_MODULES},
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


# Remove matplotlib agg warnings from generated doc when using plt.show
# From: https://github.com/sphinx-gallery/sphinx-gallery/blob/master/doc/conf.py
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

# numpydoc_show_class_members = False
