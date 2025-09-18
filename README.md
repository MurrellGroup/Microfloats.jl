# <img src="docs/src/assets/icon.svg" width="200" align="right"> Microfloats

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Microfloats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Microfloats.jl)

Microfloats is a Julia package that implements floating point types and arithmetic (through wider intermediates) for sub-8 bit floating point types, supporting arbitrary combinations of sign, exponent, and mantissa bits.

Instances of a sub-8 bit floating point type are still 8 bits wide in memory; the goal of `Microfloat` is to serve as a base for arithmetic operations and method dispatch, lending downstream packages a good abstraction for doing bitpacking and hardware acceleration.

## Usage

Along with the types already exported by Microfloats, we can also create our own types by passing the number of sign, exponent, and mantissa bits to the `Microfloat` type constructor. For example, one can recreate the `Float8` and `Float8_4` types exported by Float8s.jl:

```julia
using Microfloats

const Float8 = Microfloat{1,3,4,IEEE_754_like}
const Float8_4 = Microfloat{1,4,3,IEEE_754_like}

# creating a sawed-off Float16 (BFloat8?) becomes trivial:
const Float8_5 = Microfloat{1,5,2,IEEE_754_like}

# unsigned variants:
const UFloat7 = Microfloat{0,3,4,IEEE_754_like}
const UFloat7_4 = Microfloat{0,4,3,IEEE_754_like}
const UFloat7_5 = Microfloat{0,5,2,IEEE_754_like}
```

### Microscaling (MX)

Microfloats implements the E4M3, E5M2, E2M3, E3M2, E2M1, and E8M0 types from the [Open Compute Project Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). These are exported as `MX_E4M3`, `MX_E5M2`, `MX_E2M3`, `MX_E3M2`, `MX_E2M1`, and `MX_E8M0`, respectively, with most of these using saturated arithmetic (no Inf or NaN), and a different encoding for the types that do have NaNs.

For INT8, see `FixedPointNumbers.Q1f6`.

> [!NOTE]
> MX types may not be fully MX compliant, but efforts have been and continue to be made to adhere to the specification. See issues with the [![MX-compliance](https://img.shields.io/github/labels/MurrellGroup/Microfloats.jl/mx-compliance)](https://github.com/MurrellGroup/Microfloats.jl/labels/mx-compliance) label.

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
