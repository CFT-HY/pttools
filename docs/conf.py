# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os.path
import sys
import typing as tp

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(dir_path))

from pttools.logging import setup_logging
setup_logging()

# Create a directory for static files to avoid a warning when building.
os.makedirs(os.path.join(dir_path, "_static"), exist_ok=True)

# -- Project information -----------------------------------------------------

project = 'PTtools'
_authors = [
    "Mark Hindmarsh",
    "Mulham Hijazi",
    "Mudhahir Al-Ajmi",
    "Mika Mäki",
    "Chloe Gowling",
    "Daniel Cutting"
]
author = f"{', '.join(_authors[:-1])} & {_authors[-1]}"
copyright = f"2015-2022, {author}"
version = "0.0.1"
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    # Automatic documentation for Python code
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
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

# Automatic section labeling produces duplicated labels. This silences the warnings from those.
# https://github.com/sphinx-doc/sphinx/issues/7728
# https://github.com/sphinx-doc/sphinx/issues/7697
suppress_warnings = ["autosectionlabel.*"]

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

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
}
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


# -- Other -------------------------------------------------------------------

# Sphinx 6.0 will require base URLs and caption strings to contain exactly one "%s",
# and all other "%" need to be escaped as "%%".
extlinks: tp.Dict[str, tp.Tuple[str, tp.Optional[str]]] = {
    # Hindmarsh articles
    "gw_ssm": ("https://arxiv.org/abs/1304.2433%s", "Hindmarsh et al., 2014%s"),
    "ssm": ("https://arxiv.org/abs/1608.04735%s", "Hindmarsh et al., 2018%s"),
    "gw_pt_ssm": ("https://arxiv.org/abs/1909.10040%s", "Hindmarsh et al., 2019%s"),
    "notes": ("https://arxiv.org/abs/2008.09136%s", "Hindmarsh et al., 2021%s"),
    # Other articles
    "borsanyi_2016": ("https://arxiv.org/abs/1606.07494%s", "Borsanyi et al., 2016%s"),
    "giese_2020": ("https://arxiv.org/abs/2004.06995%s", "Giese et al., 2020%s"),
    "giese_2021": ("https://arxiv.org/abs/2010.09744%s", "Giese et al., 2021%s"),
    # Other
    "aof_grant": (
        "https://akareport.aka.fi/ibi_apps/WFServlet?IBIF_ex=x_hakkuvaus2&CLICKED_ON=&HAKNRO1=%s&UILANG=en&TULOSTE=HTML",
        "Academy of Finland grant %s"
    ),
    "issue": ("https://github.com/hindmars-org/pttools/issues/%s", "issue %s"),
    "rel_hydro_book": (
        "https://doi.org/10.1093/acprof:oso/9780198528906.001.0001%s",
        "Relativistic hydrodynamics, Rezzolla, Zanotti, 2013%s"),
    "ssm_repo": ("https://bitbucket.org/hindmars/sound-shell-model/src/master/%s", "sound-shell-model/%s")
}
intersphinx_mapping: tp.Dict[str, tp.Tuple[str, tp.Optional[str]]] = {
    "cobaya": ("https://cobaya.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    # "yappi": ("https://yappi.readthedocs.io/en/latest/", None),
}
linkcheck_allowed_redirects = {
    r"https://bitbucket\.org/*": r"https://id\.atlassian\.com/*",
    r"https://www.helsinki\.fi/": r"https://www.helsinki\.fi/en",
}
# The authentication info could be set up to work on the CI build
# https://docs.github.com/en/actions/reference/authentication-in-a-workflow
# linkcheck_auth = []
linkcheck_ignore = [
    # This website does not allow crawlers
    # r"https://academic.oup.com/book/*",
    # The private Bitbucket repos will also return 404 without authentication
    r"https://bitbucket.org/hindmars/sound-shell-model/*",
    # This link redirects to a site that does not allow crawlers
    f"https://doi.org/10.1093/acprof:oso/9780198528906.001.0001",
    # The project repository will return 404 without authentication until it's published.
    r"https://github\.com/hindmars-org/pttools/*",
]
if "GITHUB_ACTIONS" in os.environ:
    linkcheck_ignore += [
        r"https://akareport\.aka\.fi/ibi_apps/WFServlet*",
        r"https://gtr\.ukri\.org/*",
        r"https://stfc\.ukri\.org/",
    ]

linkcheck_timeout = 5
linkcheck_workers = 10

sphinx_gallery_conf = {
    "backreferences_dir": "gen_modules/backreferences",
    "compress_images": ("images", "thumbnails"),
    "doc_module": ("pttools", ),
    "examples_dirs": os.path.join(os.path.dirname(dir_path), "examples"),
    "gallery_dirs": "auto_examples",
    "ignore_pattern": r"(__init__\.py|utils\.py)",
    # "image_srcset": ["2x"],
    # "matplotlib_animations": True,
    "reference_url": {
        "pttools": None,
    },
    "show_memory": True,
}
autosummary_generate = True
