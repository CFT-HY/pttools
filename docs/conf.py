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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a directory for static files to avoid a warning when building.
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static"), exist_ok=True)

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


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    # Automatic documentation for Python code
    "sphinx.ext.autodoc",
    # Automatic labeling for documentation sections
    "sphinx.ext.autosectionlabel",
    # External links
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    # Mathematics rendering
    "sphinx.ext.mathjax",
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

# -- Other -------------------------------------------------------------------

extlinks: tp.Dict[str, tp.Tuple[str, tp.Optional[str]]] = {
    # Hindmarsh articles
    "gw_ssm": ("https://arxiv.org/abs/1304.2433", "Hindmarsh et al., 2014"),
    "ssm": ("https://arxiv.org/abs/1608.04735", "Hindmarsh et al., 2018"),
    "gw_pt_ssm": ("https://arxiv.org/abs/1909.10040", "Hindmarsh et al., 2019"),
    "notes": ("https://arxiv.org/abs/2008.09136", "Hindmarsh et al., 2021"),
    # Other articles
    "borsanyi_2016": ("https://arxiv.org/abs/1606.07494", "Borsanyi et al., 2016"),
    "giese_2020": ("https://arxiv.org/abs/2004.06995", "Giese et al., 2020"),
    "giese_2021": ("https://arxiv.org/abs/2010.09744", "Giese et al., 2021"),
    # Other
    "aof_grant": (
        "https://akareport.aka.fi/ibi_apps/WFServlet?IBIF_ex=x_hakkuvaus2&CLICKED_ON=&HAKNRO1=%s&UILANG=en&TULOSTE=HTML",
        "Academy of Finland grant %s"
    ),
    "issue": ("https://github.com/hindmars/pttools/issues/%s", "issue %s"),
    "ssm_repo": ("https://bitbucket.org/hindmars/sound-shell-model/src/master/", "sound-shell-model/%s")
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
    # The project repository will return 404 without authentication until it's published.
    r"https://github\.com/hindmars/pttools/*",
]
if "GITHUB_ACTIONS" in os.environ:
    linkcheck_ignore += [
        r"https://akareport\.aka\.fi/ibi_apps/WFServlet*",
        r"https://stfc\.ukri\.org/",
    ]

linkcheck_timeout = 5
linkcheck_workers = 10
