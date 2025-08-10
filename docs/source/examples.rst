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