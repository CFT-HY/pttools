"""Plotting utilities for testing"""

import os
import os.path

import matplotlib.pyplot as plt

FIG_FORMATS = ("pdf", "png", "svg")


def save_fig_multi(fig: plt.Figure, path: str, makedirs: bool = True):
    """Save the figure in multiple formats"""
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    for fmt in FIG_FORMATS:
        fig.savefig(f"{path}.{fmt}")
