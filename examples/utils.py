import os.path
import sys

import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG_DIR, exist_ok=True)

# Import Giese code from Mika's thesis project
MSC2_PYTHON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "msc-thesis2",
    "msc2-python"
)
if os.path.exists(MSC2_PYTHON_PATH):
    sys.path.append(MSC2_PYTHON_PATH)


def save(fig: plt.Figure, path: str):
    has_extension = "." in path
    if not os.path.isabs(path):
        path = os.path.join(FIG_DIR, path)
        if has_extension:
            fig.savefig(path)
        else:
            fig.savefig(f"{path}.eps")
            fig.savefig(f"{path}.png")


def save_and_show(fig: plt.Figure, path: str):
    save(fig, path)
    plt.show()
