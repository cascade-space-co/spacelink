Core Modules
============

The **Core** subpackage provides fundamental calculations, units, constants, and validation logic for radio frequency applications. These modules form the foundation of SpaceLink and are designed to be used both directly and by the higher-level **PHY** subpackage.

**Available Modules:**

* **units**: Defines unit types, conversion functions, and utilities for working with RF quantities including wavelength/frequency conversions, dB/linear conversions, VSWR calculations, and unit enforcement decorators
* **antenna**: Antenna calculations including polarization loss, dish gain, polarization utilities, radiation patterns, and spherical interpolation with support for circular and linear polarization
* **pattern_io**: I/O operations for antenna radiation patterns with support for various file formats
* **noise**: Noise power and density calculations, conversions between noise temperature, noise figure/factor, and Eb/N0 ↔ C/N0 relationships
* **path**: Path loss calculations for radio communications, including free space path loss, spreading loss, and aperture loss calculations
* **ranging**: Two-way sequential and PN radiometric ranging, including acquisition probability/time calculations and power allocation among carrier, ranging, and data signals

.. toctree::
   :maxdepth: 2

   units
   antenna
   pattern_io
   noise
   path
   ranging
