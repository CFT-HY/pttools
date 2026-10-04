"""Exporting and importing models, bubbles and spectra as HDF5 files.

The exportable fields of each class are defined in
:py:mod:`pttools.models.export`, :py:mod:`pttools.bubble.export`,
:py:mod:`pttools.ssm.export` and :py:mod:`pttools.omgw0.export`,
and the field selection utilities in :py:mod:`pttools.utils.fields`.
The objects of other :py:class:`~pttools.utils.fields.Extractable` classes, e.g. the spectra of other libraries,
can be exported to tables of their own by setting their :py:attr:`~pttools.utils.fields.Extractable.TABLE`.

.. code-block:: python

    from pttools.export import Exporter, Importer, Table

    with Exporter("spectra.h5") as exporter:
        exporter.add_many(spectra)

    with Importer("spectra.h5", verify=True) as importer:
        params = importer.read_scalars(Table.SPECTRA_Y)
        omgw0_h2 = importer.read(Table.SPECTRA_Y, "omgw0_h2")
"""

from pttools.utils.fields import Extractable, Field, Fields, FieldShape, FieldSpec, FieldType, Preset

from .checksum import *
from .exporter import *
from .importer import *
from .records import *
