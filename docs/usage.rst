Installation and usage
======================

Until the PTtools repository is made public, these downloads require SSH authentication
just like any GitHub repository cloning over SSH.
Please request access to the repository from prof. Hindmarsh.

If you're just using PTtools for your project,
:ref:`installation using a package manager such as pip <With pip>` is recommended.
However, if you're developing PTtools itself, you should :ref:`work on a cloned repository <Local development>`.

With pip
--------
Installing PTtools within a
`virtual environment <https://docs.python.org/3/tutorial/venv.html>`_
is highly recommended, as this enables the use of the latest versions
of its dependencies without affecting other software installed on the same computer.
The virtual environment can be created with the following commands.

.. code-block:: bash

  # The --upgrade-deps argument is not supported by Python versions older than 3.9
  # and can be left out.
  python3 -m venv --upgrade-deps venv
  source ./venv/bin/activate

Once the virtualenv is activated with the commands above, you can install PTtools from the Git repository with pip.
A PyPI package will be available later, once PTtools is made open source.

.. code-block:: bash

  # The "[performance]" installs additional libraries
  # such as icc-rt and tbb for better performance with Numba
  pip3 install --upgrade "pttools[performance] @ git+ssh://git@github.com/hindmars/pttools.git"

Alternatively you can install the development version.

.. code-block:: bash

  # The "[performance]" installs additional libraries
  # such as icc-rt and tbb for better performance with Numba
  pip3 install --upgrade "pttools[performance] @ git+ssh://git@github.com/hindmars/pttools.git@dev"

If you get an error about missing ``cmake``, you have to install it manually.
On Debian- and Ubuntu-based systems this can be done with the following commands.
Once ``cmake`` is installed, run the pip installation above again.

.. code-block:: bash

  sudo apt-get update
  sudo apt-get install cmake

With conda
----------
PTtools does not yet have a
`conda-forge <https://conda-forge.org/>`_
package, as those are quite cumbersome to maintain.
If you'd like to have one, please make a feature request in the
:issue:`issue tracker <>`.

With Docker
-----------
Before PTtools is published as open source, the direct builds from Git require,
that Docker can find your SSH keys.
This can be accomplished by running Docker
`without sudo <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.

Stable version

.. code-block:: bash

  docker build "git@github.com:hindmars/pttools.git#main" --tag pttools

Development version

.. code-block:: bash

  docker build "git@github.com:hindmars/pttools.git#dev" --tag pttools:dev

Local development version

.. code-block:: bash

  git clone git@github.com:hindmars/pttools.git
  cd pttools
  git checkout dev
  docker build . --tag pttools:dev

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
Please see the Slurm job script templates in the tests folder.

Numba errors
------------
If you get a Numba error when using PTtools, please do the following.

- Check that the parameters you give to the PTtools functions are of the types specified by their type hints.
- Upgrade Numba and other libraries to the latest versions.
  PTtools should support a wide range of Numba versions, but some of the older versions may have subtle bugs that
  apply only to a few versions.
  Incompatibilities between the versions of Numba, NumPy and llvmlite may also cause errors.
- If the issue persists, please create an issue in the :issue:`issue tracker <>`.

Parallelism
-----------
The most of the computation is serial, but some steps benefit significantly from parallel CPU resources.
These include:

- :meth:`pttools.ssmtools.calculators.sin_transform()`
- :meth:`pttools.ssmtools.spectrum.spec_den_v()`
