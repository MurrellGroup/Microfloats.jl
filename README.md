# Microfloats

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Microfloats.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Microfloats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Microfloats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Microfloats.jl)

Microfloats is a Julia package that implements types and arithmetic (through wider intermediates) for sub-8 bit floating points, supporting arbitrary combinations of sign, exponent, and mantissa (significand) bits.

Instances of a sub-8 bit floating point type are still 8 bits wide in memory; the goal of `Microfloat` is to serve as a base for arithmetic operations and method dispatch, lending downstream packages a good abstraction for doing bitpacking and hardware acceleration.

## Usage

Define your own primitive type with the macro:

```julia
using Microfloats

@microfloat MyE5M2 sign=1 exponent=5 significand=2 nonfinite=IEEE
```

Or the hand-written equivalent:

```julia
primitive type MyE5M2 <: Microfloat{1,5,2} 8 end
Microfloats.non_finite_behavior(::Type{MyE5M2}) = IEEE
```

## Overflow policy

Each `Microfloat` type carries its overflow policy as a compile-time trait,
baked in at declaration via `@microfloat`. There is no runtime override —
to convert with different semantics, declare a second type with the same
bit layout and the alternate policy, and `reinterpret` between them.

- `SAT` saturates out-of-range finite values to `±floatmax(T)`.
- `OVF` maps them to the type's sentinel: `±Inf` for `IEEE`, `NaN` for
  `NanOnlyAllOnes` (unavailable for `FiniteOnly`).

Default rule: `OVF` if the type has any non-finite sentinel, else `SAT`
(forced for `FiniteOnly`). This matches cutile-python / OCP strict
semantics. The shipped policies:

| Type                                              | NonFiniteBehavior | Overflow |
| ------------------------------------------------- | ----------------- | -------- |
| `Float8_E5M2`, `Float8_E4M3`, `Float8_E3M4`       | `IEEE`            | `OVF`    |
| `Float8_E4M3FN`, `Float8_E8M0FNU`                 | `NanOnlyAllOnes`  | `OVF`    |
| `Float6_E2M3FN`, `Float6_E3M2FN`, `Float4_E2M1FN` | `FiniteOnly`      | `SAT`    |

For PyTorch/Triton-style saturating E4M3FN, declare a twin:

```julia
@microfloat Float8_E4M3FN_SAT exponent=4 significand=3 nonfinite=NanOnlyAllOnes overflow=SAT
```

## Installation

```julia
using Pkg
Pkg.add("Microfloats")
```

## See also

- [FixedPointNumbers.jl](https://github.com/JuliaMath/FixedPointNumbers.jl)
- [MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)
- [DLFP8Types.jl](https://github.com/chengchingwen/DLFP8Types.jl)
- [Float8s.jl](https://github.com/JuliaMath/Float8s.jl)
