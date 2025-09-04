# Microfloats

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Microfloats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Microfloats.jl)

Microfloats is a Julia package that implements floating point types and arithmetic for sub-8 bit floating point types, supporting arbitrary combinations of sign, exponent, and mantissa bits.

Instances of a sub-8 bit floating point type are still 8 bits wide in memory; the goal of `Microfloat` is to serve as a base for arithmetic operations and method dispatch, lending downstream packages a good abstraction for doing bitpacking and hardware acceleration.

## Usage

We can recreate the `Float8` and `Float8_4` types exported by Float8s.jl through the `Microfloat` type constructor, which takes the number of sign, exponent, and mantissa bits as arguments:

```julia
using Microfloats

const Float8 = Microfloat(1, 3, 4)
const Float8_4 = Microfloat(1, 4, 3)

# creating a sawed-off Float16 (BFloat8?) becomes trivial:
const Float8_5 = Microfloat(1, 5, 2)
```

### Microscaling (MX)

Microfloats implements the E4M3, E5M2, E2M3, E3M2, E2M1, and E8M0 types from the [Open Compute Project Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), with most of these using saturated arithmetic (no infinities), and different bit layouts for NaNs. These are exported as `MX_E4M3`, `MX_E5M2`, `MX_E2M3`, `MX_E3M2`, `MX_E2M1`, and `MX_E8M0`, respectively.

For INT8, see `FixedPointNumbers.Q1f6`.

> [!WARNING]
> MX types may not yet be fully OCP compliant. See issues with the [![MX-compliance](https://img.shields.io/github/labels/MurrellGroup/Microfloats.jl/mx-compliance)](https://github.com/MurrellGroup/Microfloats.jl/labels/mx-compliance) label.

Since Microfloats.jl only implements the primitive types, microscaling itself may be done with [Microscaling.jl](https://github.com/MurrellGroup/Microscaling.jl), which includes quantization and bitpacking.

## Installation

```julia
using Pkg
Pkg.Registry.add(url="https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.add("Microfloats")
```

## See also

- [Microscaling.jl](https://github.com/MurrellGroup/Microscaling.jl)
- [FixedPointNumbers.jl](https://github.com/JuliaMath/FixedPointNumbers.jl)
- [MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)
- [DLFP8Types.jl](https://github.com/chengchingwen/DLFP8Types.jl)
- [Float8s.jl](https://github.com/JuliaMath/Float8s.jl)
