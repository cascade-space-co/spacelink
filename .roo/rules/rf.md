---
description: RF engineering principles and conventions
globs: src/spacelink/core/*.py
---
# RF Engineering Principles

## Units and Conversions
- Adding dB to dBW is an allowed operation
- Adding dB to dB is an allowed operation
- When attenuation or loss are specified in dB, they are always positive
- Gain in dB can be positive or negative

## Path Loss and Propagation
- Free space path loss should be separated into two distinct components:
  1. Spreading loss 1/(4πd²) - the spherical spreading of the wave
  2. Aperture loss λ²/(4π) - the wavelength-dependent effective aperture
- This separation provides more physical insight and flexibility in calculations

## Friis Transmission Equation
- The complete equation relates received power to transmitted power: Pr = Pt × Gt × Gr × (λ/4πd)²
- When implementing, maintain clear separation between antenna gains and path losses
- Always use consistent units (typically meters for distance, Hz for frequency)

## Noise Calculations
- Noise can be represented as either noise figure (dB) or noise temperature (K)
- For cascaded systems, calculate using Friis' noise formula with linear values before converting to dB
- Reference temperature for noise calculations is typically 290K
