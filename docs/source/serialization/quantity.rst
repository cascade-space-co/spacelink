Quantity Serialization
======================

The **quantity** module provides ``QuantityModel`` and ``QuantityRangeModel`` —
Pydantic models for JSON-safe serialization of astropy ``Quantity`` objects.

Overview
--------

Astropy ``Quantity`` objects carry both a numeric value and a unit, but they are
not directly JSON-serializable. ``QuantityModel`` solves this by storing the value
as a Python ``float`` (scalar) or ``list[float]`` (1D array) alongside the unit as
a canonical string produced by ``q.unit.to_string()``.

.. code-block:: python

   import astropy.units as u
   from spacelink.serialization import QuantityModel

   # Serialize
   q = 1420.0 * u.MHz
   model = QuantityModel.from_astropy(q)
   print(model.model_dump())
   # {'value': 1420.0, 'unit': 'MHz'}

   # Deserialize
   q2 = model.to_astropy()
   # <Quantity 1420. MHz>

Unit Handling
-------------

**Standard units** (``"m"``, ``"Hz"``, ``"K"``, ``"W / Hz"``, ``"K / m"``,
``"dB(W)"``, etc.) are resolved via ``astropy.units.Unit()``, which handles all
composite forms automatically.

**Legacy dB short-forms** from cascade-designer JSON are handled via an explicit
alias table:

.. list-table::
   :header-rows: 1
   :widths: 20 40

   * - Stored string
     - Resolved unit
   * - ``"dB"``
     - ``u.dB(1)`` — dimensionless dB ratio
   * - ``"dBW"``
     - ``u.dB(u.W)``
   * - ``"dBm"``
     - ``u.dB(u.mW)``
   * - ``"dBHz"``
     - ``u.dB(u.Hz)``
   * - ``"dBK"``
     - ``u.dB(u.K)``
   * - ``"dB/K"``
     - ``u.dB(1 / u.K)`` — G/T quantities

The canonical astropy forms (``"dB(W)"``, ``"dB(mW)"``, etc.) pass directly to
``u.Unit()`` and do not need aliases.

``unit=None`` or an absent ``unit`` field is treated as dimensionless.

Scalar vs Array Values
----------------------

``QuantityModel.value`` is typed as ``float | list[float]``, supporting both
scalar and 1D-array-valued quantities:

.. code-block:: python

   import numpy as np
   import astropy.units as u
   from spacelink.serialization import QuantityModel

   # Scalar
   model = QuantityModel.from_astropy(10.0 * u.dB(u.W))
   # value=10.0, unit='dB(W)'

   # Array
   model = QuantityModel.from_astropy(np.array([1.0, 2.0, 3.0]) * u.m)
   # value=[1.0, 2.0, 3.0], unit='m'

The scalar path reconstructs a Python ``float`` (not a 0-d numpy array) so that
``isinstance(q.value, float)`` holds after a round-trip.

Range Values
------------

``QuantityRangeModel`` wraps two ``QuantityModel`` fields for ranges (e.g.,
frequency bands):

.. code-block:: python

   from spacelink.serialization import QuantityModel, QuantityRangeModel

   band = QuantityRangeModel(
       min=QuantityModel(value=8.0, unit="GHz"),
       max=QuantityModel(value=8.4, unit="GHz"),
   )
   print(band.model_dump())
   # {'min': {'value': 8.0, 'unit': 'GHz'}, 'max': {'value': 8.4, 'unit': 'GHz'}}

No ``min <= max`` or unit-compatibility validation is applied — callers enforce
constraints at the point of use.

API Reference
-------------

.. automodule:: spacelink.serialization.quantity
   :members:
   :undoc-members:
   :show-inheritance:
