Installation
============

There are multiple ways to install PTtools.
If you're just using PTtools for your project,
:ref:`installation using pip <With pip>` or :ref:`Docker <With Docker>` is recommended.
However, if you're developing PTtools itself, you should
:ref:`clone the repository <Local development>`.


With pip
--------
Installing PTtools within a
`virtual environment <https://docs.python.org/3/tutorial/venv.html>`_
is highly recommended, as this enables the use of the latest versions
of its dependencies without affecting other software installed on the same computer.
The virtual environment can be created with the following commands.

.. code-block:: bash

  python3 -m venv --upgrade-deps venv
  # This activates the virtual environment for the current shell session,
  # and will have to be run again for each new shell session or console window.
  source ./venv/bin/activate

Once the virtual environment is activated with the commands above,
you can install PTtools from the Git repository with pip.
A PyPI package will be available later.

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

Stable version

.. code-block:: bash

  pip3 install --upgrade "pttools-gw[numbalsoda,performance] @ git+https://github.com/CFT-HY/pttools.git"

Development version

.. code-block:: bash

  pip3 install --upgrade "pttools-gw[numbalsoda,performance] @ git+https://github.com/CFT-HY/pttools.git@dev"


With conda
----------
PTtools does not yet have a
`conda-forge <https://conda-forge.org/>`_
package, as those are quite cumbersome to maintain.
If you'd like to have one, please make a feature request in the
:issue:`issue tracker <>`.


With Docker
-----------
PTtools container has not yet been published in a container registry,
and therefore you have to build it yourself.
Once you have built the PTtools container,
you can build your own containers which use PTtools by starting their Dockerfiles with ``FROM pttools``.

Stable version

.. code-block:: bash

  docker build "https://github.com/CFT-HY/pttools.git#main" --tag pttools
  docker run -it pttools

Development version

.. code-block:: bash

  docker build "https://github.com/CFT-HY/pttools.git#dev" --tag pttools:dev
  docker run -it pttools:dev

Local development version

.. code-block:: bash

  git clone git@github.com:hindmars/pttools.git
  cd pttools
  git checkout dev
  docker build . --tag pttools:dev
  docker run -it pttools:dev


Local development
-----------------
You can set up a local development environment with the following commands.

.. code-block:: bash

  git clone git@github.com:hindmars/pttools.git
  cd pttools
  git checkout dev
  # The --upgrade-deps argument is not supported by Python versions older than 3.9
  # and can be left out.
  python3 -m venv --upgrade-deps venv
  source ./venv/bin/activate
  pip3 install -r requirements.txt -r requirements-dev.txt
  # Now you can run the unit tests to ensure that the installation was successful.
  pytest


On a cluster
------------
For running a local development installation of PTtools on a Slurm cluster,
please see the job script templates in the tests folder.


NumbaLSODA
----------
`NumbaLSODA <https://pypi.org/project/numbalsoda/>`_
is an optional dependency, which speeds up the integration of ordinary differential equations (ODE).
It's in an early stage and may require build tools such as ``cmake`` for its installation,
and it seems not to compile yet on Windows.
You can install NumbaLSODA manually with

.. code-block:: bash

  pip3 install --upgrade numbalsoda

You may also try building from the Git repository.

.. code-block:: bash

  pip3 install --upgrade "numbalsoda @ git+https://github.com/Nicholaswogan/numbalsoda.git"

If you get an error about missing ``cmake``, you have to install it manually.
On Debian- and Ubuntu-based systems this can be done with the following commands.
Once ``cmake`` is installed, run the pip installation above again.

.. code-block:: bash

  sudo apt-get update
  sudo apt-get install cmake


Numba compatibility and nested parallelism
------------------------------------------
Nested parallelism is currently disabled by default due to the difficulty
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
