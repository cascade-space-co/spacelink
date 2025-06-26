Usage Examples
==============

.. note::
   These examples use only the core functions from ``spacelink.core``. The ``components`` submodule and cascaded noise figure analysis are no longer available.

Satellite Link Budget Analysis
------------------------------

This example demonstrates how to calculate a basic satellite link budget using only the core functions provided by SpaceLink:

.. code-block:: python

   import astropy.units as u
   from spacelink.core.units import to_dB, to_linear
   from spacelink.core.noise import noise_power, temperature_to_noise_figure
   from spacelink.core.pathloss import free_space_path_loss

   # Define system parameters
   frequency = 12 * u.GHz
   distance = 36000 * u.km  # GEO orbit
   bandwidth = 36 * u.MHz
   tx_power = 20 * u.W
   tx_antenna_gain = 40 * u.dB  # Example value
   rx_antenna_gain = 50 * u.dB  # Example value
   system_noise_temp = 290 * u.K

   # Calculate free-space path loss
   fspl = free_space_path_loss(frequency, distance)
   print(f"Free-space path loss: {fspl.to(u.dB):.2f}")

   # Calculate received power using Friis equation (in dB):
   # Pr[dBW] = Pt[dBW] + Gt[dB] + Gr[dB] - Lfs[dB]
   tx_power_dbw = to_dB(tx_power.to(u.W))
   pr_dbw = tx_power_dbw + tx_antenna_gain + rx_antenna_gain - fspl.to(u.dB)
   print(f"Received power: {pr_dbw.to(u.dBW):.2f}")

   # Calculate noise power
   n0 = noise_power(bandwidth, system_noise_temp)
   n0_dbw = to_dB(n0.to(u.W))
   print(f"Noise power: {n0_dbw.to(u.dBW):.2f}")

   # Calculate carrier-to-noise ratio (C/N)
   cn_db = pr_dbw - n0_dbw
   print(f"Carrier-to-noise ratio (C/N): {cn_db.to(u.dB):.2f}")

   # Convert system noise temperature to noise figure (optional)
   nf = temperature_to_noise_figure(system_noise_temp)
   print(f"System noise figure: {nf.to(u.dB):.2f}")

Expected Output
---------------

.. code-block:: text

   Free-space path loss: 205.16 dB
   Received power: -102.15 dB(W)
   Noise power: -128.41 dB(W)
   Carrier-to-noise ratio (C/N): 26.27 dB
   System noise figure: 3.01 dB

#
# Additional examples can be added here as new core functions are introduced.
#
