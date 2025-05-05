---
description: Creating link budgets
globs: 
alwaysApply: false
---
This file is used for guidelines for creating link budgets

# Style of Link Budget
- Object oriented. Use [stage.py](mdc:src/spacelink/components/stage.py) modules wherever possible
- Avoid use of core/ modules and fuctions. Prefer modules from components/
- Forward and Return are standard names for directions
  * users sometimes specify uplink and downlink for this

## Primary Link Drivers
These parameters are the most commonly tuned and experimented with as factors driving the link margin and data throughput
- Transmit power
- Ground station (dish) size
- Symbol rate
- Coding scheme
These are the primary figures of merit for evaluating the results of the design choices above:
- Margin
- C/N0

# Definitions
- LEO = low earth orbit
- GTO = geostationary transfer orbit
- TLI = trans-lunar injection
- Near Space = LEO, GTO, Lunar, TLI, under 2M km
- Deep Space = more than 2M km

# Allocations for Space Research Services
- Near Space S-Band links
  * Forward frequencies: 2000 MHz to 2110 MHz
  * Return frequencies: 2200 MHz to 2290 MHz
  * Bandwidths: 5 MHz maximum
- Near Space X-Band links
  * Forward frequencies: 7190 MHz to 7235 MHz
  * Return frequencies: 8450 MHz to 8500 MHz

- Deep Space S-Band links
  * Forward frequencies: 2110 MHz to 2120 MHz
  * Return frequencies: 2290 MHz to 2300 MHz
  * Bandwidths: 5 MHz Maximum

- Deep Space X-Band links
  * Forward frequencies: 7145 MHz to 7190 MHz
  * Return frequencies: 8450 MHz to 8500 MHz
  * Bandwidths: 10 MHz Maximum

# Typical Structure
## Transmit
On the transmit side, the most commonly used blocks are:
- Transmitter with an output power
- Coax cable, and perhaps a splitter with loss. This can be modeled as a single gain block with a loss
- Antenna. On the ground this is a Dish, on the spacecraft this is a FixedGain
## Path
Currently only modeling simple free space path loss

## Receive


# Common Assumptions
## Spacecraft Antennas
For TT&C links, typically using S-Band, the antenna systems on spacecraft employ omnidirectional systems using patch antennas. These patterns have deep nulls, and are typically modeled as having anywhere from 0 to -20 dB of gain.

These antenna noise temperatures vary. If in LEO and Earth is in most of the field of view of one half of the antenna, then the noise temp is the average of 290K on Earth side and about 10K on the space side