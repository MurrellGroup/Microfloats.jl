# Microfloats

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Microfloats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Microfloats.jl)

Microfloats is a Julia package that implements types and arithmetic (through wider intermediates) for sub-byte floating points, supporting arbitrary combinations of sign, exponent, and significand (mantissa) bits.

Instances of a sub-8 bit floating point type are still one byte wide in memory; Microfloats serves as a base for method dispatch and a reference for arithmetic operations, lending downstream packages like [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) a useful layer of abstraction.

## Usage

Define your own primitive type with the macro:

```julia
using Microfloats

@microfloat MyE5M2 sign=1 exponent=5 significand=2 nonfinite=Microfloats.IEEE
```

or see the documentation for a list of predefined types.

## Installation

```julia
using Pkg
Pkg.add("Microfloats")
```

## See also

- [FixedPointNumbers.jl](https://github.com/JuliaMath/FixedPointNumbers.jl)
- [Float8s.jl](https://github.com/JuliaMath/Float8s.jl)
- [DLFP8Types.jl](https://github.com/chengchingwen/DLFP8Types.jl)
- [MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)
