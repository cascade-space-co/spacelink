Antenna Module
===================

Polarization Loss
-----------------

The polarization loss between two antennas with given axial ratios is calculated using the standard formula for polarization mismatch:

.. math::
   PLF = \frac{1}{2} + \frac{1}{2} \frac{4 \gamma_1 \gamma_2 - (1-\gamma_1^2)(1-\gamma_2^2)}{(1+\gamma_1^2)(1+\gamma_2^2)}

where:

* :math:`\gamma_1` and :math:`\gamma_2` are the voltage axial ratios (linear, not dB)
* PLF is the polarization loss factor (linear)

The polarization loss in dB is then:

.. math::
   L_{pol} = -10 \log_{10}(PLF)

For circular polarization, the axial ratio is 0 dB, and for linear polarization, it is >40 dB.

Dish Gain
---------

The gain of a parabolic dish antenna is given by:

.. math::
   G = \eta \left(\frac{\pi D}{\lambda}\right)^2

where:

* :math:`\eta` is the efficiency factor (typically 0.55 to 0.70)
* :math:`D` is the diameter of the dish
* :math:`\lambda` is the wavelength

In decibels:

.. math::
   G_{dB} = 10\log_{10}(\eta) + 20\log_{10}(D) + 20\log_{10}(f) + 20\log_{10}\left(\frac{\pi}{c}\right)

.. automodule:: spacelink.core.antenna
   :members:
   :undoc-members:
   :show-inheritance:
