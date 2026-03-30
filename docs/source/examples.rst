Usage Examples
==============

This section provides practical examples demonstrating how to use the SpaceLink library for various space communications analysis tasks.

Jupyter Notebooks
-----------------

The examples are provided as Jupyter notebooks in the ``notebooks/`` directory:

* **Link Budget Analysis** (``examples/leo_link_budget.ipynb``): Complete LEO satellite link budget calculation
* **Modulation and Coding** (``examples/modulation_and_coding.ipynb``): Performance analysis of different modulation schemes

Theory and Background
---------------------

For theoretical background and detailed explanations, see the notebooks in ``notebooks/theory/``:

* **Antenna Theory** (``theory/antenna.ipynb``): Antenna modeling and polarization analysis
* **Noise Analysis** (``theory/noise.ipynb``): Thermal noise and SNR calculations
* **Path Analysis** (``theory/path.ipynb``): Free space path loss and link budget fundamentals
* **Ranging** (``theory/ranging.ipynb``): Timing and ranging calculations
* **Units and Conversions** (``theory/units.ipynb``): Unit handling and conversions

Running Examples
----------------

To run the examples, first ensure you have the development dependencies installed:

.. code-block:: bash

   poetry install --with dev

Then start Jupyter notebook using Poetry:

.. code-block:: bash

   poetry run jupyter notebook


All examples use the SpaceLink library and include detailed explanations of the underlying theory and implementation.

Serialization Examples
----------------------

These examples show how to use ``QuantityModel`` and ``QuantityRangeModel`` to
serialize and deserialize astropy quantities for use in APIs, databases, or
configuration files.

Scalar quantity round-trip
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import astropy.units as u
   from spacelink.serialization import QuantityModel

   # Create a quantity
   eirp = 47.3 * u.dB(u.W)

   # Serialize to a dict (JSON-safe)
   model = QuantityModel.from_astropy(eirp)
   data = model.model_dump()
   # {'value': 47.3, 'unit': 'dB(W)'}

   # Deserialize back to an astropy Quantity
   eirp2 = QuantityModel.model_validate(data).to_astropy()
   # <Quantity 47.3 dB(W)>

Array quantity round-trip
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import astropy.units as u
   from spacelink.serialization import QuantityModel

   gains = np.array([30.0, 31.5, 32.0]) * u.dB(1)

   model = QuantityModel.from_astropy(gains)
   data = model.model_dump()
   # {'value': [30.0, 31.5, 32.0], 'unit': 'dB'}

   gains2 = QuantityModel.model_validate(data).to_astropy()
   # <Quantity [30. , 31.5, 32. ] dB>

Frequency range
^^^^^^^^^^^^^^^

.. code-block:: python

   from spacelink.serialization import QuantityModel, QuantityRangeModel

   x_band = QuantityRangeModel(
       min=QuantityModel(value=8.0, unit="GHz"),
       max=QuantityModel(value=8.4, unit="GHz"),
   )

   data = x_band.model_dump()
   # {'min': {'value': 8.0, 'unit': 'GHz'}, 'max': {'value': 8.4, 'unit': 'GHz'}}

   band2 = QuantityRangeModel.model_validate(data)
   low = band2.min.to_astropy()   # <Quantity 8. GHz>
   high = band2.max.to_astropy()  # <Quantity 8.4 GHz>

Legacy dB unit strings
^^^^^^^^^^^^^^^^^^^^^^

When reading JSON produced by older cascade-designer data, short-form dB strings
are automatically resolved:

.. code-block:: python

   from spacelink.serialization import QuantityModel

   # cascade-designer stores G/T as "dB/K"
   gt = QuantityModel(value=-25.0, unit="dB/K")
   q = gt.to_astropy()
   # <Quantity -25. dB(1 / K)>

   # Canonical astropy form works identically
   gt2 = QuantityModel(value=-25.0, unit="dB(1/K)")
   q2 = gt2.to_astropy()
   # <Quantity -25. dB(1 / K)>