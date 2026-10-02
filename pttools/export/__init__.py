"""Exporting and importing models, bubbles and spectra as HDF5 files, e.g. for training machine learning models.

The exportable fields of each class are defined in
:py:mod:`pttools.models.export`, :py:mod:`pttools.bubble.export`,
:py:mod:`pttools.ssm.export` and :py:mod:`pttools.omgw0.export`,
and the field selection utilities in :py:mod:`pttools.utils.fields`.

.. code-block:: python

    from pttools.export import Exporter, Importer, Table

    with Exporter("spectra.h5") as exporter:
        exporter.add_many(spectra)

    with Importer("spectra.h5", verify=True) as importer:
        params = importer.read_scalars(Table.SPECTRA)
        omgw0_h2 = importer.read(Table.SPECTRA, "omgw0_h2")
"""

from pttools.utils.fields import Field, Fields, FieldShape, FieldSpec, FieldType, Preset

from .checksum import *
from .exporter import *
from .importer import *
from .records import *
