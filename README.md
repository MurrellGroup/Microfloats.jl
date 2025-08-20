# Microfloats

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Microfloats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Microfloats.jl)

Microfloats is a Julia package that implements floating point types and arithmetic for sub-8 bit floating point types, supporting arbitrary combinations of sign, exponent, and mantissa bits.

Instantiated sub-8 bit floating point types are still 8 bits wide; the goal of `Microfloat` is to serve as a base for arithmetic operations on these types, allowing for downstream packages to implement e.g. hardware-accelerated and bit-packed operations on these types.

## Usage

We can recreate the `Float8` and `Float8_4` types exported by Float8s.jl through the `Microfloat` type constructor, which takes the number of sign, exponent, and mantissa bits as arguments:

```julia
using Microfloats

const Float8 = Microfloat(1, 3, 4)
const Float8_4 = Microfloat(1, 4, 3)

# creating a sawed-off Float16 (BFloat8?) becomes trivial:
const Float8_5 = Microfloat(1, 5, 2)
```

### MX format

Microfloats additionally implements the `E4M3`, `E5M2`, `E2M3`, `E3M2`, `E2M1`, and `E8M0` formats from the [Open Compute Project Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), with most of these using saturated arithmetic (no infinities), and different bit layouts for NaNs. These can be constructed by passing an additional `:MX` argument to the `Microfloat` constructor:

```julia
const E4M3 = Microfloat(1, 4, 3, :MX)
const E5M2 = Microfloat(1, 5, 2, :MX)
const E2M3 = Microfloat(1, 2, 3, :MX)
const E3M2 = Microfloat(1, 3, 2, :MX)
const E2M1 = Microfloat(1, 2, 1, :MX)
const E8M0 = Microfloat(0, 8, 0, :MX)
```

## Installation

```julia
using Pkg
Pkg.Registry.add(url="https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.add("Microfloats")
```

## See also

- [MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)
- [DLFP8Types.jl](https://github.com/chengchingwen/DLFP8Types.jl)
- [Float8s.jl](https://github.com/JuliaMath/Float8s.jl)
