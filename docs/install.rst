Installation
============

PTtools is a Python library and can therefore be installed with the same tools and commands as other Python libraries.
If you're using PTtools for your project,
:ref:`installation using pip <With pip>` or :ref:`Docker <With Docker>` is recommended.
If you're developing PTtools itself, you should
:ref:`clone the repository <Local development>`.


With pip
--------
To install PTtools with pip, you need to have Python installed.
You can install Python from the `official website <https://www.python.org/>`_,
or by using the package manager of your operating system (e.g. apt).

Installing PTtools within a
`virtual environment <https://docs.python.org/3/tutorial/venv.html>`_
is highly recommended, as this enables the use of the latest versions
of its dependencies without affecting other software installed on the same computer.
The virtual environment can be created with the following commands.

.. code-block:: bash

  python3 -m venv --upgrade-deps venv
  # The following command activates the virtual environment for the current shell session,
  # and will have to be run again for each new shell session or console window.
  source ./venv/bin/activate

Once the virtual environment is activated with the commands above,
you can install PTtools with pip.

Stable version from PyPI:

.. code-block:: bash

  pip3 install --upgrade pttools-gw[numbalsoda,performance]

From the Git repository. For the main branch, you can omit the "@BRANCH_NAME".

.. code-block:: bash

  pip3 install --upgrade "pttools-gw[numbalsoda,performance] @ git+https://github.com/CFT-HY/pttools.git@BRANCH_NAME"


The ``[numbalsoda]`` flag installs the optional
`NumbaLSODA <https://pypi.org/project/numbalsoda/>`_
ordinary differential equation (ODE) solver library,
but it may not build on all platforms, especially Windows.
Therefore if you get any build errors,
please remove the ``[numbalsoda]`` flag and try the PTtools installation again.
You can then have a look at the
:ref:`NumbaLSODA` section of the installation instructions.

The ``[performance]`` flag installs additional libraries such as
`icc-rt <https://pypi.org/project/icc-rt/>`_
and
`tbb <https://pypi.org/project/tbb/>`_
for better performance with Numba.


With conda
----------
PTtools does not yet have a
`conda-forge <https://conda-forge.org/>`_
package, as those are quite cumbersome to maintain.
If you'd like to have one, please make a feature request in the
:issue:`issue tracker <>`.


With Docker
-----------
PTtools is
[available on Docker Hub](https://hub.docker.com/r/cfthy/pttools).
You can build your own container which uses PTtools by starting its Dockerfile with ``FROM cfthy/pttools:main``.

From Docker Hub:

.. code-block:: bash

  docker pull cfthy/pttools:main
  docker run -it cfthy/pttools:main

From the Git repository. For the main branch, you can omit the "#BRANCH_NAME".

.. code-block:: bash

  docker build "https://github.com/CFT-HY/pttools.git#BRANCH_NAME" --tag pttools:BRANCH_NAME
  docker run -it pttools:BRANCH_NAME

Local development version

.. code-block:: bash

  git clone git@github.com:CFT-HY/pttools.git
  cd pttools
  git checkout BRANCH_NAME
  docker build . --tag pttools:BRANCH_NAME
  docker run -it pttools:BRANCH_NAME


Local development
-----------------
You can set up a local development environment with the following commands.

.. code-block:: bash

  git clone git@github.com:CFT-HY/pttools.git
  cd pttools
  python3 -m venv --upgrade-deps venv
  source ./venv/bin/activate
  pip3 install -r requirements.txt -r requirements-dev.txt -r ./docs/requirements.txt
  # Now you can run the unit tests to ensure that the installation was successful.
  pytest

You can build the documentation locally with the following commands.

.. code-block:: bash

  cd docs
  make html

With conda
^^^^^^^^^^
If you want to set up PTtools for local development and install the dependencies with conda,
you can do so with the following commands.

.. code-block:: bash

  git clone git@github.com:CFT-HY/pttools.git
  cd pttools
  conda env create --file environment.yml
  conda activate pttools

On a cluster
------------
For running a local development installation of PTtools on a Slurm cluster,
please see the job script templates in the tests folder.


NumbaLSODA
----------
`NumbaLSODA <https://pypi.org/project/numbalsoda/>`_
is an optional dependency, which speeds up the integration of ordinary differential equations (ODE).
You can install NumbaLSODA manually with

.. code-block:: bash

  pip3 install --upgrade numbalsoda

You may also try building NumbaLSODA from the Git repository.

.. code-block:: bash

  pip3 install --upgrade "numbalsoda @ git+https://github.com/Nicholaswogan/numbalsoda.git"

Due to the low-level design of NumbaLSODA,
if a pre-built binary is not available for your platform,
you need to have CMake and compiler tools for C++ and Fortran installed.
The steps for installing these depend on your platform.

Linux
^^^^^
If you get an error about missing ``cmake`` or ``gfortran``, you have to install them manually.
On Debian- and Ubuntu-based systems this can be done with the following commands.
Once the packages are installed, run the pip installation above again.

.. code-block:: bash

  sudo apt-get update
  sudo apt-get install cmake gfortran

macOS
^^^^^
If you get an error on macOS about missing ``gfortran``,
you have to install the GCC compiler tools, which include ``gfortran``.
You can do this e.g. with `Homebrew <https://brew.sh/>`_ or `MacPorts <https://www.macports.org/>`_.
To install ``gfortran`` with MacPorts, please first install MacPorts and then run the following commands.

.. code-block:: bash

  sudo port install gcc14
  sudo port select --set gcc mp-gcc14

Windows
^^^^^^^
Installing NumbaLSODA on Windows requires build tools such as
`Build Tools for Visual Studio <https://visualstudio.microsoft.com/downloads/?q=build+tools>`_
and a Fortran compiler such as
`Intel Fortran Compiler <https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler-download.html>`_.
In addition to these, you may also have to install CMake manually.


Numba compatibility and nested parallelism
------------------------------------------
Nested parallelism, which means running parallel code within parallel code,
is currently disabled by default due to the difficulty
in setting up OpenMP and TBB on cluster environments and macOS.

Some parts of the code such as
:meth:`pttools.ssmtools.spectrum.spec_den_gw_scaled`
can use nested parallelism to optimally use all available CPU resources.
This requires that either OpenMP or Intel TBB is installed,
as Numba's integrated workqueue backend does not support nested parallelism.
Therefore if you get the error

.. code-block::

  Terminating: Nested parallel kernel launch detected,
  the workqueue threading layer does not supported nested parallelism.
  Try the TBB threading layer.

when running a program that uses PTtools, or the error
``Fatal Python error: Aborted``
when running pytest,
please install either OpenMP or Intel TBB (or both).
You can verify that the installation works by running the command ``numba --sysinfo``
and checking the contents of the section ``Threading Layer Information``.
If you can't get the threading backends working,
you can disable the nested parallelism by setting the environment variable
``NUMBA_NESTED_PARALLELISM=0`` before importing PTtools.
For example, this command should work for the
:ref:`Local development` version without the threading libraries:

.. code-block:: bash

  NUMBA_NESTED_PARALLELISM=0 pytest
