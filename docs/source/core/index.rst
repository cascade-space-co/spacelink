Core Modules
============

The Core subpackage provides fundamental calculations, units, constants, and validation logic for radio frequency applications. These modules form the foundation of SpaceLink and are designed to be used both directly and by the higher-level Components subpackage.

Key Modules
-----------

* **units**: Defines unit types, conversion functions, and utilities for working with RF quantities
* **antenna**: Antenna calculations including dish gain and polarization loss
* **noise**: Functions for noise calculations, including conversions between noise figure and temperature
* **path**: Path loss calculations for radio communications, including free space path loss
* **ranging**: Functions for ranging and distance calculations


.. toctree::
   :maxdepth: 2

   units
   antenna
   noise
   path
   ranging
