Welcome to SpaceLink's documentation!
=====================================

SpaceLink is a comprehensive Python library for radio frequency (RF) calculations, designed specifically for space communications and RF system analysis. Developed by `Heliosphere Network <https://www.heliospherenetwork.com>`_, SpaceLink provides tools for antenna modeling, RF chain analysis, link budget calculations, and noise analysis.

Key Features
------------

* **Antenna Modeling**: Calculate antenna gain, beamwidth, and polarization effects for various antenna types including parabolic dishes
* **RF System Analysis**: Model complete RF chains with cascaded components including amplifiers, attenuators, and filters
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

   git clone https://github.com/Heliosphere-Network/spacelink.git
   cd spacelink
   poetry install

Basic Usage
^^^^^^^^^^^

Here's a simple example of calculating free space path loss:

.. code-block:: python

   import astropy.units as u
   from spacelink.core.path import free_space_path_loss

   # Calculate path loss for a 2.4 GHz signal over 1000 km
   distance = 1000 * u.km
   frequency = 2.4 * u.GHz

   loss = free_space_path_loss(distance, frequency)
   print(f"Free space path loss: {loss}")

Library Structure
-----------------

SpaceLink is organized into two main subpackages:

* **Core**: Fundamental calculations, units, constants, and validation logic with minimal dependencies
* **Components**: Object-oriented models for RF components and systems, building on the core functionality

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   examples
   math_examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`