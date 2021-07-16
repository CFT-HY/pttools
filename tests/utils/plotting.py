import matplotlib.pyplot as plt

FIG_FORMATS = ("pdf", "png", "svg")


def save_fig_multi(fig: plt.Figure, path: str):
    for fmt in FIG_FORMATS:
        fig.savefig(f"{path}.{fmt}")
