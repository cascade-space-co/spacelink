PHY Modules
===========

The **PHY** subpackage provides physical-layer constructs for link analysis, enabling comprehensive modeling of communication system performance. These modules build upon the **Core** subpackage to provide higher-level abstractions for link mode definitions and performance analysis.

**Available Modules:**

* **mode**: Link mode definitions combining modulation schemes and coding chains (FEC, framing, etc.) with support for concatenated codes and interleaving
* **performance**: Performance curves (Eb/N0 vs. error rate) and coding gain calculations with interpolation support for BER, WER, and FER metrics
* **registry**: A registry system that loads modes and performance data from YAML files for easy configuration management and data persistence

**Example:**

Load a predefined BPSK mode with concatenated coding (outer RS(255,223) with interleaver depth 5, inner convolutional rate 1/2) from the built-in registry:

.. code-block:: python

   from spacelink.phy.registry import Registry
   from spacelink.phy.performance import ErrorMetric

   # Load built-in modes and performance data
   reg = Registry()
   reg.load()  # uses packaged YAML under spacelink/phy/data

   # Print available mode IDs
   print(sorted(reg.modes.keys()))

   # Fetch the desired mode by ID
   mode_id = "CCSDS_TM_RS255223_I5_CONV7-12_BPSK"
   mode = reg.modes[mode_id]

   # Inspect the loaded configuration
   print(mode.modulation.name)
   print(mode.coding.rate)
   print(mode.info_bits_per_symbol)

.. toctree::
   :maxdepth: 2

   mode
   performance
   registry


