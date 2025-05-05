Usage Examples
==============

This page provides practical examples of using SpaceLink for common RF and space communication tasks.

Satellite Link Budget Analysis
------------------------------

This example calculates a complete link budget for a satellite communication system:

.. code-block:: python

   import astropy.units as u
   from spacelink.components.antenna import Dish
   from spacelink.components.stage import TransmitAntenna, ReceiveAntenna, Path
   from spacelink.components.transmitter import Transmitter
   from spacelink.components.demodulator import Demodulator
   from spacelink.components.mode import Mode

   # Define system parameters
   frequency = 12 * u.GHz
   distance = 36000 * u.km  # GEO orbit

   # Create transmitter (satellite)
   tx_power = 20 * u.W
   tx = Transmitter(power=tx_power, noise_temperature=290 * u.K)

   # Create transmit antenna (satellite)
   tx_dish = Dish(diameter=1.2 * u.m, efficiency=0.65)
   tx_antenna = TransmitAntenna(antenna=tx_dish)
   tx_antenna.input = tx

   # Create path between satellite and ground
   space_path = Path(distance=distance)
   space_path.input = tx_antenna

   # Create receive antenna (ground station)
   rx_dish = Dish(diameter=3.7 * u.m, efficiency=0.7)
   rx_antenna = ReceiveAntenna(antenna=rx_dish)
   rx_antenna.input = space_path

   # Create demodulator (ground station)
   demod = Demodulator(conversion_loss=2 * u.dB, noise_temperature=290 * u.K)
   demod.input = rx_antenna

   # Define modulation mode
   qpsk = Mode(
       name="QPSK",
       modulation="QPSK",
       coding_scheme="LDPC",
       code_rate=0.5,
       required_ebno=4.5 * u.dB,
       implementation_loss=2 * u.dB
   )

   # Calculate received signal power
   rx_power = demod.get_processed_signal().power
   print(f"Received power: {rx_power.to(u.dBm)}")

   # Calculate system noise temperature
   sys_noise_temp = demod.get_processed_signal().noise_temperature
   print(f"System noise temperature: {sys_noise_temp.to(u.K)}")

   # Calculate link margin
   link_margin = demod.calculate_link_margin(frequency, qpsk, bandwidth=36 * u.MHz)
   print(f"Link margin: {link_margin.to(u.dB)}")

Cascaded Noise Figure Analysis
------------------------------

This example analyzes the noise performance of a cascaded RF chain:

.. code-block:: python

   import astropy.units as u
   import matplotlib.pyplot as plt
   import numpy as np
   from spacelink.components.source import Source
   from spacelink.components.stage import GainBlock, Attenuator
   from spacelink.components.sink import Sink

   # Create components
   source = Source()
   lna = GainBlock(gain=20 * u.dB, noise_figure=1.2 * u.dB)
   filter1 = Attenuator(attenuation=3 * u.dB)
   amp2 = GainBlock(gain=15 * u.dB, noise_figure=3.5 * u.dB)
   filter2 = Attenuator(attenuation=2 * u.dB)
   mixer = GainBlock(gain=-6 * u.dB, noise_figure=8 * u.dB)
   if_amp = GainBlock(gain=30 * u.dB, noise_figure=2.5 * u.dB)
   sink = Sink()

   # Connect the chain
   lna.input = source
   filter1.input = lna
   amp2.input = filter1
   filter2.input = amp2
   mixer.input = filter2
   if_amp.input = mixer
   sink.input = if_amp

   # Calculate cascaded properties at different stages
   frequency = 2.4 * u.GHz
   stages = [lna, filter1, amp2, filter2, mixer, if_amp]
   stage_names = ["LNA", "Filter 1", "Amp 2", "Filter 2", "Mixer", "IF Amp"]

   # Calculate cumulative gain and noise figure at each stage
   gains = []
   noise_figures = []

   for stage in stages:
       gains.append(stage.cascaded_gain(frequency).to(u.dB).value)
       noise_figures.append(stage.cascaded_noise_figure(frequency).to(u.dB).value)

   # Plot results
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

   ax1.plot(stage_names, gains, 'o-', linewidth=2)
   ax1.set_ylabel('Cascaded Gain (dB)')
   ax1.set_title('Cascaded RF Chain Analysis')
   ax1.grid(True)

   ax2.plot(stage_names, noise_figures, 'o-', linewidth=2, color='red')
   ax2.set_ylabel('Cascaded Noise Figure (dB)')
   ax2.set_xlabel('Stage')
   ax2.grid(True)

   plt.tight_layout()
   plt.show()

   # Print final results
   print(f"Total gain: {gains[-1]:.2f} dB")
   print(f"Total noise figure: {noise_figures[-1]:.2f} dB")
