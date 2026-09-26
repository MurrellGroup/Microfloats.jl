```@meta
DocTestSetup = :(using Microfloats)
```

# Conversion

Every conversion reduces to one function, [`Microfloats.cvt`](@ref). Constructors,
`convert` and broadcasts only resolve defaults and call it, so there is one
place to specialize a conversion and one place for a device backend to
override it.

| Direction | Scalar | Vector |
|---|---|---|
| into a `Microfloat` | `cvt(T, x, mode, policy)` | `cvt(SVector{N,T}, xs, mode, policy)`, `cvt(NVector{T,N}, xs, mode, policy)` |
| out of a `Microfloat` | `cvt(F, x)` | `cvt(SVector{N,F}, xs)` |

Narrowing takes a rounding mode and an [overflow policy](@ref "Overflow policies")
as positional arguments, so methods can dispatch on them. Widening into
`Float16`, `BFloat16`, `Float32` or `Float64` takes neither: every `Microfloat`
is exactly representable in `BFloat16`.

```jldoctest
julia> using Microfloats: cvt, SAT

julia> cvt(Float8_E4M3FN, 1000f0, RoundNearest, SAT)
Float8_E4M3FN(448.0)

julia> cvt(Float32, Float8_E4M3FN(1.5))
1.5f0
```

```@docs
Microfloats.cvt
Microfloats.cvt_generic
Microfloats.cvt_twiddle
Microfloats.cvt_lanes
Microfloats.WideFloat
```

## Vectors of lanes

Two static vector types hold `N` values of one format:

- `SVector{N,T}` (StaticArrays) gives each value its own byte.
- `NVector{T,N}` (BitPacking) packs the values densely, `bitwidth(T)` bits each.

For 8-bit formats the two have the same bits. They differ for narrower
formats, and both layouts occur in hardware: a 4-bit pair is one byte, while a
6-bit pair is two bytes with each value in the low six bits of its own byte.
The aliases name the layout NVIDIA's packed types (`__nv_fp8x2_e4m3`,
`__nv_fp6x2_e2m3`, `__nv_fp4x2_e2m1`, …) and the PTX `cvt` instructions use:

| Alias | Type | Storage |
|---|---|---|
| `Float8x2_E4M3FN`, `Float8x4_E4M3FN` | `NVector{Float8_E4M3FN,N}` | `UInt16`, `UInt32` |
| `Float8x2_E5M2`, `Float8x4_E5M2` | `NVector{Float8_E5M2,N}` | `UInt16`, `UInt32` |
| `Float8x2_E8M0FNU`, `Float8x4_E8M0FNU` | `NVector{Float8_E8M0FNU,N}` | `UInt16`, `UInt32` |
| `Float6x2_E2M3FN`, `Float6x4_E2M3FN` | `SVector{N,Float6_E2M3FN}` | one byte per value |
| `Float6x2_E3M2FN`, `Float6x4_E3M2FN` | `SVector{N,Float6_E3M2FN}` | one byte per value |
| `Float4x2_E2M1FN`, `Float4x4_E2M1FN` | `NVector{Float4_E2M1FN,N}` | `UInt8`, `UInt16` |
| `Float16x2`, `BFloat16x2`, … | `SVector{N,Float16}`, `SVector{N,BFloat16}` | two bytes per value |

Densely packed 6-bit storage, as in memory, is `NVector{Float6_E2M3FN,N}`.

Both vector types construct from any static vector of lanes, and widen back:

```jldoctest
julia> using Microfloats: SVector, NVector, SAT

julia> xs = SVector(0.5f0, 1f0, -6f0, 100f0);

julia> packed = NVector{Float4_E2M1FN,4}(xs; overflow=SAT);

julia> sizeof(packed)
2

julia> Tuple(SVector{4,Float32}(packed))
(0.5f0, 1.0f0, -6.0f0, 6.0f0)
```

Packing is layout only. The packed narrowing form converts through the
`SVector` form and packs, and packed widening sources unpack to a tuple of
lanes, so a specialization of the unpacked form serves both.

## Specializing a conversion

Optimized implementations are ordinary methods on the funnel, chosen by
dispatch:

1. [`Microfloats.cvt_generic`](@ref) is the bit-level reference path.
   [`Microfloats.cvt_twiddle`](@ref) computes the same results branch-free and
   is the default for `Float32` sources, and so for every source that
   converts to `Float32` first.
2. [`Microfloats.@cvt_table`](@ref) compiles the generic results for one pair
   of microfloat types into a bit-twiddle or a lookup table. Every built-in
   pair has one.
3. Hand-written methods.
4. Device overrides in package extensions.

`@cvt_table` runs every source bit pattern through the generic path and fits
the results with linear pieces `(i << k) + c` over the source bits `i`. Exact
widenings, such as `Float4_E2M1FN` to `Float8_E4M3FN`, are one piece over the
source normals and one per binade of source subnormals; saturated and NaN
results are constant pieces. A combination with few pieces becomes a
branch-free twiddle, which vectorizes and so also serves packed sources and
destinations of any length:

```julia
# Float4_E2M1FN => Float8_E4M3FN, any rounding mode and overflow policy
i = reinterpret(UInt8, x) & 0x07
t = ifelse(i >= 0x02, (i << 2) + 0x30, ifelse(i >= 0x01, (i << 3) + 0x28, 0x00))
t |= (reinterpret(UInt8, x) & 0x08) << 4
```

Everything else is a lookup. [`Microfloats.max_twiddle_cost`](@ref) sets the
cutoff; the CUDA extension lowers it, since on the device a cached table load
beats all but the shortest twiddles.

Widening is generated the same way, without registration: `@microfloat` fits
the conversions of each new type into `Float16`, `BFloat16` and `Float32` over
the destination's bits. Normals are one linear piece. Zero and all subnormals
share one piece evaluated on the FPU: the subnormal significand placed under
the exponent of ``2^{1-\text{bias}}``, minus ``2^{1-\text{bias}}``, is the
exact value, normalized. `Float64` extends the `Float32` result.

```@docs
Microfloats.@cvt_table
Microfloats.max_twiddle_cost
```

## CUDA

With CUDACore loaded, kernels and broadcasts lower conversions to the native
PTX `cvt` instructions where the compile target has them, and run the generic
path otherwise. The choice is made when the kernel is compiled.

| Conversion | Operands | Target |
|---|---|---|
| to `Float8_E4M3FN`, `Float8_E5M2` | `Float32`, `Float16` | sm_89 and newer |
| to `Float8_E4M3FN`, `Float8_E5M2` | `BFloat16` | sm_100 and newer, arch or family target |
| to `Float6_E2M3FN`, `Float6_E3M2FN`, `Float4_E2M1FN` | `Float32`, `Float16`, `BFloat16` | sm_100 and newer, arch or family target |
| to `Float8_E8M0FNU` (`RoundToZero`, `RoundUp`) | `Float32`, `BFloat16` | sm_100 and newer, arch or family target |
| from `Float8_E4M3FN`, `Float8_E5M2` | `Float16` | sm_89 and newer |
| from any of the six formats | `Float16`, `BFloat16`, `Float32` | sm_100 and newer, arch or family target |

`Float8_E8M0FNU` has no native `Float16` form. `Float32` widens through
`BFloat16`.

Hardware narrowing always saturates, so native narrowing applies to the `SAT`
policy with `RoundNearest` (or the two listed modes for `Float8_E8M0FNU`);
everything else takes the branch-free `cvt_twiddle` path. Vector forms use
one instruction per two lanes for any even `N`. Native and generic results agree bit for bit,
apart from NaN payloads.
