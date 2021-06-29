Usage of PTtools
==================

With pip
--------
TODO

With conda
----------
TODO

With Docker
-----------
TODO

On a cluster
------------
Please see the job script templates in the tests folder.

Local development
-----------------
You can set up a local development enviroment with the following commands.

.. code-block:: bash

  python -m venv venv
  source ./venv/bin/activate
  pip install -r requirements.txt

Now you can run the unit tests with ``pytest``


Parallelism
-----------
The most of the computation is serial, but some steps benefit significantly from parallel CPU resources.
These include:

- ``ssmtools.calculators.sin_transform()``
