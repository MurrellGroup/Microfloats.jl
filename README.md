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

Conversions take a rounding mode and an overflow policy, and work on scalars, on static vectors with one value per byte, and on densely packed vectors:

```julia
using Microfloats
using Microfloats: SVector, NVector, SAT

Float8_E4M3FN(1000f0, RoundToZero; overflow=SAT)                    # Float8_E4M3FN(448.0)

packed = NVector{Float4_E2M1FN,4}(SVector(0.5f0, 1f0, -6f0, 100f0)) # two bytes
SVector{4,Float32}(packed)                                          # 0.5, 1.0, -6.0, 6.0
```

All of them reduce to one dispatchable function, `Microfloats.cvt`. With CUDACore loaded, kernels and broadcasts lower it to native conversion instructions on GPUs that have them. See [Conversion](https://murrellgroup.github.io/Microfloats.jl/dev/conversion/) in the documentation.

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
