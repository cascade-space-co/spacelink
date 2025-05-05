# Overview  
SpaceLink was created to eliminate the use of spreadsheets when doing link budget analysis, but it is capable of so much more than that. It is a single library that allows the user to perform cascaded analysis of both transmit and receive amplifier chains, calculate link margin for both standard modulation and coding schemes as well as deep space ranging modes, and ultimately design spacecraft more efficiently.

Furthermore, use of this library streamlines dynamic link modeling, making it easy for users to sweep parameters like range and symbol rate, and also to perform trade studies for spacecraft RF architectures.

# Core Features
## core module
- Implements functions that abstract equations and formulas from RF engineering

## components 
- Object-oriented library for cascaded analysis using components/ modules
- Chainable objects
- Demodulator object encapsulates all 
- Serialization to/from YAML
- Standard directional terms: `Forward` (e.g., ground to space) and `Return` (e.g., space to ground). (`uplink`/`downlink` are common user terms)

# User Experience  
## User Persona: RF Engineer who Codes
This is an RF (radio frequency) engineer who writes code in Python or C++ and no longer relies on spreadsheets for all calculations.

This person is willing to do analysis in jupyter notebooks or write python scripts, but they are not necessarily experienced software engineers.

# Technical Architecture  
- System components
- Object-oriented structure:
  - Transmit Chain (Source -> Stages -> TxAntenna)
  - Path (Path Stage)
  - Receive Chain (RxAntenna -> Stages -> Sink/Demodulator)
- Primary Link Drivers (Key parameters for analysis):
  - Transmit power
  - Ground station antenna size (e.g., Dish diameter)
  - Symbol rate
  - Modulation and Coding scheme
- Key Figures of Merit (Results):
  - Link Margin
  - C/N0 (Carrier-to-Noise-Density Ratio)
- Data models
- APIs and integrations
- Infrastructure requirements

# Development Roadmap  

**Current State (Features Already Implemented):**
- Core `Stage` components (`Source`, `GainBlock`, `Attenuator`, `TransmitAntenna`, `ReceiveAntenna`, `Path`, basic `Sink`).
- Stage chaining via `input` property.
- Core calculations (`gain`, `cascaded_gain`, `noise_figure`, `cascaded_noise_figure`, `output` signal propagation).
- `FixedGain` and `Dish` antenna models.
- Basic free space `Path` loss.
- Serialization/Deserialization (YAML support).
- Core `modcod` definitions (`DataMode`, required Eb/N0 lookups - *RangingMode status TBC*).

## Phase 1 (MVP): Implement Demodulator & Link Margin Calculation
**Goal:** Enable end-to-end link margin calculation within a Jupyter notebook.

**Key Features/Components:**
- **Develop `Demodulator` Stage:**
  - Acts as the final stage (likely replaces the basic `Sink`).
  - Takes `Signal` (power, noise temperature) from the preceding stage as input.
  - Accepts parameters defining the communication mode (e.g., an instance of `DataMode` or `RangingMode`), symbol rate, and potentially required bandwidths.
  - Calculates C/N0 (Carrier-to-Noise-Density ratio) from the input signal.
  - Calculates Eb/N0 (Energy per bit to noise density ratio) using C/N0 and bit rate (derived from symbol rate and code rate via `DataMode`).
  - Retrieves required Eb/N0 for data modes using `DataMode`.
  - Calculates Data Link Margin (`Actual Eb/N0 - Required Eb/N0`).
  - Integrates logic (from `core.ranging`?) to calculate Ranging Link Margin based on C/N0 and ranging mode parameters (`RangingMode`?).
  - Provides clear methods like `cn0()`, `ebno()`, `data_margin()`, `ranging_margin()`.
- **Verify `core.modcod` & `core.ranging`:** Ensure the necessary functions/classes within these core modules for required Eb/N0, code rates, bit rates, and ranging calculations are implemented and correct. Address any issues found in `TestRangingMode` or `TestDataMode`.
- **Documentation/Examples:** Update notebook examples to demonstrate building a full link ending with the `Demodulator` and calculating C/N0 and link margins.

## Phase 2: Advanced Modeling & Usability
**Goal:** Increase model fidelity and improve user workflow after MVP is achieved.
*(Features listed below can be prioritized and potentially broken into smaller phases)*

**Key Features/Components:**
- **Antenna Models:** Support for more complex antenna models (e.g., importing gain patterns from files).
- **Path Loss Models:** Implement advanced path loss models (e.g., atmospheric absorption, rain attenuation).
- **Workflow Tools:** Develop utilities for parameter sweeping (e.g., sweep range, power, symbol rate) and plotting results (e.g., margin vs. range).
- **Documentation:** Enhance documentation with detailed examples covering link margin calculation, `modcod` usage, and advanced modeling features.
- **GUI/Web Interface:** (Lower priority) Explore options for a graphical interface for building links and visualizing results.
- **Integrations:** (Lower priority) Consider integrations with external mission analysis or simulation tools.

# Logical Dependency Chain
[Define the logical order of development:
- Which features need to be built first (foundation)
- Getting as quickly as possible to something usable/visible front end that works
- Properly pacing and scoping each feature so it is atomic but can also be built upon and improved as development approaches]

# Appendix: RF Domain Context

## Definitions
- LEO = low earth orbit
- GTO = geostationary transfer orbit
- TLI = trans-lunar injection
- Near Space = LEO, GTO, Lunar, TLI, distances under 2 million km
- Deep Space = distances over 2 million km

## Frequency Allocations for Space Research Services
### Near Space
- S-Band Forward: 2000 MHz to 2110 MHz (5 MHz max bandwidth)
- S-Band Return: 2200 MHz to 2290 MHz (5 MHz max bandwidth)
- X-Band Forward: 7190 MHz to 7235 MHz
- X-Band Return: 8450 MHz to 8500 MHz

### Deep Space
- S-Band Forward: 2110 MHz to 2120 MHz (5 MHz max bandwidth)
- S-Band Return: 2290 MHz to 2300 MHz (5 MHz max bandwidth)
- X-Band Forward: 7145 MHz to 7190 MHz (10 MHz max bandwidth)
- X-Band Return: 8400 MHz to 8450 MHz (10 MHz max bandwidth)

## Common Assumptions & Models
- **Spacecraft TT&C Antennas (S-Band):** Often omnidirectional (e.g., patch antennas) with gain modeled between 0 dB and -20 dB due to nulls. Noise temperature varies; ~150K average in LEO (considering Earth/space view).
- **Ground Antennas:** Typically modeled as `Dish` antennas.
- **Spacecraft Antennas (Non-TT&C):** Often modeled as `FixedGain`.
- **Path Model:** Currently simple free space path loss (`Path` stage).
- **Transmit Chain:** Typically includes Source power, loss stages (e.g., `GainBlock` for cable loss), and `TransmitAntenna`.
- **Receive Chain:** Typically includes `ReceiveAntenna`, gain stages (`GainBlock` for LNA), and a `Sink` or demodulator model.
