Core Modules
============

The Core subpackage provides fundamental calculations, units, constants, and validation logic for radio frequency applications. These modules form the foundation of SpaceLink and are designed to be used both directly and by the higher-level Components subpackage.

Key Modules
-----------

* **units**: Defines unit types, conversion functions, and utilities for working with RF quantities
* **antenna**: Core antenna calculations including dish gain and polarization loss
* **noise**: Functions for noise calculations, including conversions between noise figure and temperature
* **path**: Path loss calculations for radio communications, including free space path loss
* **channelcoding**: Functions for channel coding calculations, including coding gain and error correction
* **validation**: Validation decorators for ensuring correct function arguments

Unit System
-----------

SpaceLink uses Astropy's unit system to ensure all calculations are performed with proper physical units. This helps prevent errors and makes the code more readable and maintainable.

Example: Converting between noise figure and temperature
--------------------------------------------------------

.. code-block:: python

   import astropy.units as u
   from spacelink.core.noise import noise_figure_to_temperature, temperature_to_noise_figure

   # Convert a 2 dB noise figure to noise temperature
   nf = 2.0 * u.dB
   temp = noise_figure_to_temperature(nf)
   print(f"Noise temperature: {temp.to(u.K)}")

   # Convert back to noise figure
   nf_back = temperature_to_noise_figure(temp)
   print(f"Noise figure: {nf_back.to(u.dB)}")

.. toctree::
   :maxdepth: 2

   units
   antenna
   noise
   path
   channelcoding
   validation
