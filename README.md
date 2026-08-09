# Microfloats

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Microfloats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Microfloats.jl)

Microfloats is a Julia package that implements types and arithmetic (through wider intermediates) for sub-byte floating points, supporting arbitrary combinations of sign, exponent, and significand (mantissa) bits.

In ordinary arrays, or as single instances, sub-8 bit floating points are one byte wide; Microfloats provides a numerical reference for any conceivable microfloat, but also offers canonical types for common narrow data formats which can then be used downstream by packages like [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl).

## Usage

Define your own microfloat with the macro:

```julia
using Microfloats: @microfloat

@microfloat Float8_E5M2 sign=1 exponent=5 significand=2 nonfinite=Microfloats.IEEE
```

or find [predefined types](https://murrellgroup.github.io/Microfloats.jl/stable/predefined/) in the documentation.

## Installation

```julia
using Pkg
Pkg.add("Microfloats")
```

## See also

- [MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)
- [DLFP8Types.jl](https://github.com/chengchingwen/DLFP8Types.jl)
- [Float8s.jl](https://github.com/JuliaMath/Float8s.jl)
- [FixedPointNumbers.jl](https://github.com/JuliaMath/FixedPointNumbers.jl)
