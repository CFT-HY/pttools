"""Intersphinx configuration."""

type IntersphinxMapping = dict[str, tuple[str, str | None]]

INTERSPHINX_MAPPING: IntersphinxMapping = {
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
