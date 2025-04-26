Component Modules
=================

The Components subpackage provides object-oriented models for RF components and systems. These classes build on the core functionality to model complete RF chains and systems.

Key Concepts
------------

* **Source**: Components that generate signals (e.g., transmitters, oscillators)
* **Stage**: Components that process signals (e.g., amplifiers, filters, antennas)
* **Sink**: Components that consume signals (e.g., receivers, demodulators)
* **Signal**: Data structure representing RF signals with power, frequency, and other properties

RF Chain Modeling
-----------------

SpaceLink allows you to model complete RF chains by connecting sources, stages, and sinks. Each component can be configured with specific parameters, and the system automatically calculates cascaded properties like gain and noise figure.

Example: Building a simple receiver chain
-----------------------------------------

.. code-block:: python

   import astropy.units as u
   from spacelink.components.source import Source
   from spacelink.components.stage import GainBlock, Attenuator
   from spacelink.components.sink import Sink

   # Create a simple RF chain: Source -> LNA -> Attenuator -> Sink
   source = Source()
   lna = GainBlock(gain=20 * u.dB, noise_figure=1.5 * u.dB)
   attenuator = Attenuator(attenuation=6 * u.dB)
   sink = Sink()

   # Connect the components
   lna.input = source
   attenuator.input = lna
   sink.input = attenuator

   # Calculate cascaded properties at 2.4 GHz
   frequency = 2.4 * u.GHz
   cascaded_gain = attenuator.cascaded_gain(frequency)
   cascaded_nf = attenuator.cascaded_noise_figure(frequency)

   print(f"Cascaded gain: {cascaded_gain}")
   print(f"Cascaded noise figure: {cascaded_nf}")

Available Components
--------------------

* **Antenna**: Models for different antenna types (dish, fixed gain)
* **Demodulator**: Signal demodulation with configurable parameters
* **Receiver**: Complete receiver chains with gain and noise properties
* **Transmitter**: Signal sources with configurable power and modulation
* **Stage**: Various RF stages including gain blocks, attenuators, and paths

.. toctree::
   :maxdepth: 2

   antenna
   demodulator
   mode
   receiver
   signal
   sink
   source
   stage
   transmitter
