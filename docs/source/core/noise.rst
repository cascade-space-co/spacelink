Noise Module
============

Noise Power
-----------

The thermal noise power for a given bandwidth and temperature is:

.. math::
   P_n = k T B

where :math:`k` is Boltzmann's constant, :math:`T` is the noise temperature in Kelvin, and :math:`B` is the bandwidth in Hz.

Noise Power Density
-------------------

The noise power density for a given temperature is:

.. math::
   P_n = k T

where :math:`k` is Boltzmann's constant and :math:`T` is the noise temperature in Kelvin.

Noise Temperature to Noise Factor
---------------------------------

The noise factor (linear) is given by:

.. math::
   F = 1 + \frac{T}{T_0}

where :math:`T` is the noise temperature and :math:`T_0` is the reference temperature (290 K).

Noise Factor to Noise Temperature
---------------------------------

The noise temperature is given by:

.. math::
   T = (F - 1) T_0

where :math:`F` is the noise factor (linear) and :math:`T_0` is the reference temperature (290 K).

Noise Figure to Noise Temperature
---------------------------------

The conversion from noise figure (in dB) to noise temperature (in Kelvin) is done in two steps:

1. Convert noise figure (dB) to noise factor (linear):

   .. math::
      F = 10^{NF_{dB}/10}

2. Convert noise factor to noise temperature:

   .. math::
      T = (F - 1) T_0

where :math:`NF_{dB}` is the noise figure in dB and :math:`T_0` is the reference temperature (290 K).

Noise Temperature to Noise Figure
---------------------------------

The conversion from noise temperature in Kelvin to noise figure in dB is done in two steps:

1. Convert temperature to noise factor (linear):

   .. math::
      F = 1 + \frac{T}{T_0}

2. Convert noise factor to noise figure (dB):

   .. math::
      NF_{dB} = 10 \log_{10}(F)

where :math:`T_0` is the reference temperature (290 K).

.. automodule:: spacelink.core.noise
   :members:
   :undoc-members:
   :show-inheritance:
