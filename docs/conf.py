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
import os.path
import sys
import tomllib
import warnings

from sphinx_gallery.sorting import ExplicitOrder

dir_path = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.dirname(dir_path)
sys.path.insert(0, os.path.dirname(dir_path))

from pttools.logging import setup_logging
from pttools.utils.system import IS_GITHUB_ACTIONS
setup_logging()

# Create a directory for static files to avoid a warning when building.
os.makedirs(os.path.join(dir_path, "_static"), exist_ok=True)

# -- Project information -----------------------------------------------------

project = 'PTtools'
with open(os.path.join(repo_path, "AUTHORS"), "r") as file:
    _authors = file.read().splitlines()
author = f"{', '.join(_authors[:-1])} & {_authors[-1]}"
copyright = f"2015-{date.today().year}, {author}"
with open (os.path.join(repo_path, "pyproject.toml"), "rb") as file:
    version = tomllib.load(file)["project"]["version"]
release = version


# -- General configuration ---------------------------------------------------

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
        "path": os.path.join(repo_path, "pttools"),
        "destination": "gen_modules/pttools"
    },
    {
        "path": os.path.join(repo_path, "tests"),
        "destination": "gen_modules/tests"
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
extlinks: dict[str, tuple[str, str]] = {
    # The articles are ordered by publication year
    # Hindmarsh articles
    "hindmarsh_2014": ("https://arxiv.org/abs/1304.2433%s", "Hindmarsh et al., 2014%s"),
    "hindmarsh_2015": ("https://arxiv.org/abs/1504.03291%s", "Hindmarsh et al., 2015%s"),
    "hindmarsh_2017": ("https://arxiv.org/abs/1704.05871%s", "Hindmarsh et al., 2017%s"),
    "ssm": ("https://arxiv.org/abs/1608.04735%s", "Hindmarsh et al., 2018%s"),
    "gw_pt_ssm": ("https://arxiv.org/abs/1909.10040%s", "Hindmarsh et al., 2019%s"),
    "notes": ("https://arxiv.org/abs/2008.09136%s", "Hindmarsh et al., 2021%s"),
    # Other articles
    "enqvist_1992": ("https://doi.org/10.1103/PhysRevD.45.3415%s", "Enqvist et al., 1992%s"),
    "kurki-suonio_1995": ("https://arxiv.org/abs/hep-ph/9512202%s", "Kurki-Suonio & Laine, 1995%s"),
    "maggiore_1999": ("https://arxiv.org/abs/gr-qc/9909001%s", "Maggiore, 1999%s"),
    "espinosa_2010": ("https://arxiv.org/abs/1004.4187%s", "Espinosa et al., 2010%s"),
    "borsanyi_2016": ("https://arxiv.org/abs/1606.07494%s", "Borsanyi et al., 2016%s"),
    "caprini_2016": ("https://arxiv.org/abs/1512.06239%s", "Caprini et al., 2016%s"),
    "cornish_2017": ("https://arxiv.org/abs/1703.09858%s", "Cornish & Robson, 2017%s"),
    "planck_2018": ("https://arxiv.org/abs/1807.06209%s", "Planck 2018 results%s"),
    "smith_2019": ("https://arxiv.org/abs/1908.00546%s", "Smith & Caldwell, 2019%s"),
    "caprini_2020": ("https://arxiv.org/abs/1910.13125%s", "Caprini et al., 2020%s"),
    "giese_2020": ("https://arxiv.org/abs/2004.06995%s", "Giese et al., 2020%s"),
    "giese_2021": ("https://arxiv.org/abs/2010.09744%s", "Giese et al., 2021%s"),
    "gowling_2021": ("https://arxiv.org/abs/2106.05984%s", "Gowling & Hindmarsh, 2021%s"),
    "ajmi_2022": ("https://arxiv.org/abs/2205.04097%s", "Ajmi & Hindmarsh, 2022%s"),
    "cutting_2022": ("https://arxiv.org/abs/2204.03396%s", "Cutting, Vilhonen & Weir, 2022%s"),
    "ai_2023": ("https://arxiv.org/abs/2303.10171%s", "Ai et al., 2023%s"),
    "gowling_2023": ("https://arxiv.org/abs/2209.13551%s", "Gowling et al., 2023%s"),
    "lewicki_2023": ("https://arxiv.org/abs/2305.04924%s", "Lewicki et al., 2023%s"),
    "barni_2024": ("https://arxiv.org/abs/2406.01596%s", "Barni et al., 2024%s"),
    "croon_2024": ("https://arxiv.org/abs/2410.21509%s", "Croon & Weir, 2024%s"),
    "giombi_2024_cs": ("https://arxiv.org/abs/2409.01426%s", "Giombi et al., 2024%s"),
    "giombi_2024_gr": ("https://arxiv.org/abs/2307.12080%s", "Giombi & Hindmarsh, 2024%s"),
    "barni_2026": ("https://arxiv.org/abs/2510.21439%s", "Barni et al., 2026%s"),
    "bhusal_2026": ("https://arxiv.org/abs/2603.22397%s", "Bhusal et al., 2026%s"),
    "giombi_2026": ("https://arxiv.org/abs/2504.08037%s", "Giombi et al., 2026%s"),
    # Theses
    "gowling_phd": ("https://hdl.handle.net/10779/uos.23309135.v1%s", "Gowling, 2023%s"),
    "hakkinen_msc": ("https://hdl.handle.net/10138/576963%s", "Häkkinen, 2024%s"),
    "maki_msc": ("https://arxiv.org/abs/2511.20436%s", "Mäki, 2025%s"),
    # Other
    "aof_grant": (
        "https://akareport.aka.fi/ibi_apps/WFServlet?IBIF_ex=x_hakkuvaus2&CLICKED_ON=&HAKNRO1=%s&UILANG=en&TULOSTE=HTML",
        "Academy of Finland grant %s"
    ),
    "issue": ("https://github.com/CFT-HY/pttools/issues/%s", "issue %s"),
    "lisa_conventions": ("https://gitlab.esa.int/lisa-sgs/sandbox/conventions-document%s", "LISA DDPC Conventions document%s"),
    "lisa_sci_req": ("https://www.cosmos.esa.int/web/lisa/documents%s", "LISA Science Requirements Document%s"),
    "rel_hydro_book": (
        "https://doi.org/10.1093/acprof:oso/9780198528906.001.0001%s",
        "Relativistic hydrodynamics, Rezzolla, Zanotti, 2013%s"),
    "ssm_repo": ("https://bitbucket.org/hindmars/sound-shell-model/src/master/%s", "sound-shell-model/%s")
}
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
    # The project repository will return 404 without authentication until it's published.
    "https://github.com/CFT-HY/pttools/*",
    # The anchors are valid but not detected by Sphinx.
    "https://github.com/scipy/scipy/blob/v1.8.0/scipy/interpolate/fitpack/*",
]
if IS_GITHUB_ACTIONS:
    linkcheck_ignore += [
        r"https://akareport\.aka\.fi/ibi_apps/WFServlet*",
        r"https://www\.intel\.com/*",
        r"https://gtr\.ukri\.org/*",
        r"https://scicomp\.stackexchange\.com/*",
        r"https://stackoverflow\.com/*",
        r"https://stfc\.ukri\.org/",
    ]

# Timeout had to be increased from 5 to prevent errors with slow ArXiv links
linkcheck_timeout = 20
linkcheck_workers = 10

# show_memory = GITHUB_ACTIONS
show_memory = True

sphinx_gallery_conf = {
    "backreferences_dir": "gen_modules/backreferences",
    "compress_images": ("images", "thumbnails"),
    "doc_module": ("pttools", ),
    "examples_dirs": os.path.join(os.path.dirname(dir_path), "examples"),
    "filename_pattern": ".*",
    "gallery_dirs": "auto_examples",
    "ignore_pattern": r"(__init__\.py|utils\.py|p_s_scan_dev\.py|droplet|standard_model|entropy|reverse)",
    # "image_srcset": ["2x"],
    # "line_numbers": True,
    "matplotlib_animations": True,
    # Parallelism cannot be enabled simultaneously with "show_memory".
    # It may also produce errors with some IDEs:
    # https://stackoverflow.com/questions/31080829/python-error-io-unsupportedoperation-fileno
    "parallel": not show_memory,
    # "prefer_full_module": ...
    "reference_url": {
        "pttools": None,
        "tests": None,
    },
    # "run_stale_examples": True
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


# Remove matplotlib agg warnings from generated doc when using plt.show
# From: https://github.com/sphinx-gallery/sphinx-gallery/blob/master/doc/conf.py
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in multiply"
)

# numpydoc_show_class_members = False
