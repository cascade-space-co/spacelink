Units Module
============

Wavelength
----------

The relationship between wavelength and frequency is given by:

.. math::
   \lambda = \frac{c}{f}

where:

* :math:`c` is the speed of light (299,792,458 m/s)
* :math:`f` is the frequency in Hz


to_dB
-----

The conversion to decibels is done using:

.. math::
   \text{X}_{\text{dB}} = \text{factor} \cdot \log_{10}(x)

The result will have units of dB(input_unit), e.g. dBW, dBK, dBHz, etc. For dimensionless input, the result will have unit dB.

to_linear
---------

The conversion from decibels to a linear (dimensionless) ratio is done using:

.. math::
   x = 10^{\frac{\text{X}_{\text{dB}}}{\text{factor}}}

where:

* :math:`\text{X}_{\text{dB}}` is the value in decibels
* :math:`\text{factor}` is 10 for power quantities, 20 for field quantities

Return Loss to VSWR
-------------------

The conversion from return loss in decibels to voltage standing wave ratio (VSWR) is done using:

.. math::
   \text{VSWR} = \frac{1 + |\Gamma|}{1 - |\Gamma|}

where:

* :math:`|\Gamma|` is the magnitude of the reflection coefficient
* :math:`|\Gamma| = 10^{-\frac{\text{RL}}{20}}`
* :math:`\text{RL}` is the return loss in dB

VSWR to Return Loss
-------------------

The conversion from voltage standing wave ratio (VSWR) to return loss in decibels is done using:

.. math::
   \text{RL} = -20 \log_{10}\left(\frac{\text{VSWR} - 1}{\text{VSWR} + 1}\right)

where:

* :math:`\text{VSWR}` is the voltage standing wave ratio
* :math:`\text{RL}` is the return loss in dB

Mismatch Loss
-------------

Mismatch loss quantifies power lost from reflections at an interface. It is calculated using:

.. math::
   \text{ML} = -10 \log_{10}(1 - |\Gamma|^2)

where:

* :math:`|\Gamma|` is the magnitude of the reflection coefficient
* :math:`|\Gamma| = 10^{-\frac{\text{RL}}{20}}`
* :math:`\text{RL}` is the return loss in dB

.. automodule:: spacelink.core.units
   :members:
   :undoc-members:
   :show-inheritance:
