Core Modules
============

The Core subpackage provides fundamental calculations, units, constants, and validation logic for radio frequency applications. These modules form the foundation of SpaceLink and are designed to be used both directly and by the higher-level Components subpackage.

Key Modules
-----------

* **units**: Defines unit types, conversion functions, and utilities for working with RF quantities
* **antenna**: Antenna calculations including polarization loss, dish gain, polarization utilities, radiation patterns, and spherical interpolation
* **noise**: Noise power and density calculations, conversions between noise temperature, noise figure/factor, and Eb/N0 ↔ C/N0
* **path**: Path loss calculations for radio communications, including free space path loss
* **ranging**: Two-way sequential and PN radiometric ranging, including acquisition probability/time and power allocation among carrier, ranging, and data


.. toctree::
   :maxdepth: 2

   units
   antenna
   noise
   path
   ranging
