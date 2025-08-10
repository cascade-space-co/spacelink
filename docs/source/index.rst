Welcome to SpaceLink's documentation!
=====================================

SpaceLink is a comprehensive Python library for radio frequency (RF) calculations,
designed specifically for space communications and RF system analysis.
Developed by `Cascade Space <https://cascade.space>`_,
SpaceLink provides tools for antenna modeling, RF chain analysis,
link budget calculations, and noise analysis.
The source code is available on `GitHub <https://github.com/cascade-space-co/spacelink>`_.

Key Features
------------

* **Antenna Modeling**: Calculate antenna gain, beamwidth, and polarization effects for various antenna types including parabolic dishes
* **RF System Analysis**: Model complete RF chains with cascaded elements including amplifiers, attenuators, and filters
* **Link Budget Calculations**: Perform comprehensive analysis of radio communication links between spacecraft and ground stations
* **Noise Calculations**: Calculate system noise temperature, noise figure, and related parameters
* **Space Communications**: Built-in support for satellite link analysis with path loss and atmospheric effects
* **Unit-Aware Calculations**: All calculations use proper units through the Astropy units system
* **Physical Layer Support**: Link mode definitions with modulation, coding, and performance curves for comprehensive link analysis

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

View the source code at the `GitHub Repository <https://github.com/cascade-space-co/spacelink>`_.

Library Structure
-----------------

SpaceLink is organized into two main subpackages:

* **Core**: Fundamental calculations, units, constants, and validation logic with minimal dependencies
* **PHY**: Physical-layer definitions including modulation/coding link modes, performance curves, and a registry backed by YAML data

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   core/index
   phy/index
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`