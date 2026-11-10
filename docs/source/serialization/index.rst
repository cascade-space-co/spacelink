Serialization Modules
=====================

The **Serialization** subpackage provides JSON-safe round-trip serialization for
astropy ``Quantity`` objects. It is the bridge between spacelink's unit-aware
calculations and any system that stores or transmits RF parameters as JSON —
REST APIs, databases, configuration files, inter-service messages.

The core problem it solves: astropy ``Quantity`` objects are not JSON-serializable.
``QuantityModel`` is a Pydantic model that captures both the numeric value and the
unit string, serializes cleanly to ``{"value": 1420.0, "unit": "MHz"}``, and
reconstructs a full astropy ``Quantity`` on the way back out.

**Available Modules:**

* **quantity**: ``QuantityModel`` and ``QuantityRangeModel`` — scalar and array-valued
  quantity serialization with full unit round-trip support including all standard
  astropy units and legacy dB short-forms (dBW, dBm, dBHz, dBK, dB/K)

.. toctree::
   :maxdepth: 2

   quantity
