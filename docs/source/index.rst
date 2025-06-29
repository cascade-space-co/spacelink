Welcome to SpaceLink's documentation!
=====================================

SpaceLink is a comprehensive Python library for radio frequency (RF) calculations,
designed specifically for space communications and RF system analysis.
Developed by `Cascade Space <https://cascade.space>`_,
SpaceLink provides tools for antenna modeling, RF chain analysis,
link budget calculations, and noise analysis.

Key Features
------------

* **Antenna Modeling**: Calculate antenna gain, beamwidth, and polarization effects for various antenna types including parabolic dishes
* **RF System Analysis**: Model complete RF chains with cascaded elements including amplifiers, attenuators, and filters
* **Link Budget Calculations**: Perform comprehensive analysis of radio communication links between spacecraft and ground stations
* **Noise Calculations**: Calculate system noise temperature, noise figure, and related parameters
* **Space Communications**: Built-in support for satellite link analysis with path loss and atmospheric effects
* **Unit-Aware Calculations**: All calculations use proper units through the Astropy units system

Getting Started
---------------

Installation
^^^^^^^^^^^^

SpaceLink can be installed using pip:

.. code-block:: bash

   pip install spacelink

For development:

.. code-block:: bash

   git clone https://github.com/cascade-space-co/spacelink.git
   cd spacelink
   poetry install

Library Structure
-----------------
* **Core**: Fundamental calculations, units, constants, and validation logic with minimal dependencies

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`