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
copyright = '2015-2021, Mark Hindmarsh, Mudhahir Al-Ajmi, Danny Bail & Mika Mäki'
author = 'Mark Hindmarsh, Mudhahir Al-Ajmi, Danny Bail, Jacky Lindsay, Mike Soughton & Mika Mäki'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Automatic documentation for Python code
    # https://www.sphinx-doc.org/en/master/usage/quickstart.html#autodoc
    "sphinx.ext.autodoc",
    # External links
    # https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    # Mathematics rendering
    # https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax
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
    "undoc-members": True
}
autodoc_typehints = "description"

# -- Other -------------------------------------------------------------------

extlinks: tp.Dict[str, tp.Tuple[str, tp.Optional[str]]] = {
    "gw_ssm_article": ("https://link.aps.org/doi/10.1103/PhysRevLett.112.041301", None),
    "notes": ("https://scipost.org/10.21468/SciPostPhysLectNotes.24", None),
    "ssm_article": ("https://link.aps.org/doi/10.1103/PhysRevLett.120.071301", None),
}
intersphinx_mapping: tp.Dict[str, tp.Tuple[str, tp.Optional[str]]] = {
    "cobaya": ("https://cobaya.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None)
}
