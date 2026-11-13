r"""
Viewing the dataset of spectra
==============================

Explore a dataset created by :ref:`sphx_glr_auto_examples_dataset_save.py` interactively in a web browser
with the viewer of :py:mod:`pttools.plot.dataset.app`.

The viewer has a slider for each parameter of the dataset, with a step for each of its stored values,
and a selector for the nucleation type.
It shows the stored fluid profile of the bubble and the spectrum of the selected parameter values,
and the stored scalar values of the model, the bubble and the spectrum.
Only the selected bubble and spectrum are read from the file,
so the viewer can be used also for the large dataset.

The viewer needs Dash, which is installed with ``uv sync --all-extras`` or ``pip install "pttools-gw[dashboard]"``.
By default, the first HDF5 file in the ``data`` directory of the repository is loaded.
As the dataset is not available when building the documentation, the example is skipped by Sphinx-Gallery.
Run it from the root directory of the repository with e.g.

.. code-block:: bash

    python -m examples.dataset.view
    python -m examples.dataset.view --path /scratch/const_cs_large.h5 --port 8051

and open the printed address, by default ``http://127.0.0.1:8050``, in a browser.
If the viewer is running on a remote computer, forward the port to your computer first with e.g.

.. code-block:: bash

    ssh -L 8050:localhost:8050 user@host

The parameters of the models are detected automatically,
see :py:func:`pttools.export.index.detect_axes`, and can be given with ``--model-params``.
"""

import argparse
from pathlib import Path

from examples.dataset.load import default_path
from pttools.docs.examples import is_sphinx_gallery
from pttools.plot.dataset.app import DEFAULT_HOST, DEFAULT_PORT, run


def main(
        path: Path | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        debug: bool = False,
        verify: bool = False,
        model_params: list[str] | None = None) -> None:
    """Serve the viewer of a dataset until interrupted.

    :param path: path of the HDF5 file. Defaults to the first HDF5 file in the ``data`` directory.
    :param host: address to listen to
    :param port: port to listen to
    :param debug: whether to enable the debugging tools of Dash
    :param verify: whether to verify the file against its SHA-256 checksum file
    :param model_params: names of the parameters that identify the models. If None, they are detected automatically.
    """
    run(
        default_path() if path is None else Path(path),
        host=host, port=port, debug=debug, verify=verify, model_params=model_params)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Explore a dataset interactively in a web browser")
    parser.add_argument("--path", type=Path, help="path of the HDF5 file (default: the first one in data/)")
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"address to listen to (default: {DEFAULT_HOST}, i.e. only this computer)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen to (default: {DEFAULT_PORT})")
    parser.add_argument("--debug", action="store_true", help="enable the debugging tools of Dash")
    parser.add_argument("--verify", action="store_true", help="verify the SHA-256 checksum of the file")
    parser.add_argument(
        "--model-params", nargs="+", metavar="NAME",
        help="names of the parameters that identify the models (default: detected automatically)")
    return parser.parse_args()


# The dataset is not available when building the documentation.
if __name__ == "__main__" and not is_sphinx_gallery():
    _args = parse_args()
    main(
        path=_args.path, host=_args.host, port=_args.port, debug=_args.debug, verify=_args.verify,
        model_params=_args.model_params)
