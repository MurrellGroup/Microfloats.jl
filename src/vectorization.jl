using BitPacking: NVector
using StaticArrays: SVector, StaticArray

# 16-bit

const Float16x2 = SVector{2,Float16}
const Float16x4 = SVector{4,Float16}

const BFloat16x2 = SVector{2,BFloat16}
const BFloat16x4 = SVector{4,BFloat16}

# 8-bit

const Float8x2_E4M3FN = NVector{Float8_E4M3FN,2}
const Float8x4_E4M3FN = NVector{Float8_E4M3FN,4}

const Float8x2_E5M2 = NVector{Float8_E5M2,2}
const Float8x4_E5M2 = NVector{Float8_E5M2,4}

const Float8x2_E8M0FNU = NVector{Float8_E8M0FNU,2}
const Float8x4_E8M0FNU = NVector{Float8_E8M0FNU,4}

# 6-bit

const Float6x2_E2M3FN = NVector{Float6_E2M3FN,2}
const Float6x2_E3M2FN = NVector{Float6_E3M2FN,2}

const Float6x4_E2M3FN = NVector{Float6_E2M3FN,4}
const Float6x4_E3M2FN = NVector{Float6_E3M2FN,4}

# 4-bit

const Float4x2_E2M1FN = NVector{Float4_E2M1FN,2}
const Float4x4_E2M1FN = NVector{Float4_E2M1FN,4}

# ───────────────────────── vector conversion funnel ──────────────────────────

"""
    cvt_lanes(::Type{NVector{T,N}}, xs::NTuple{N,Any}, mode, policy) -> NVector{T,N}

Reference lanewise implementation of the vector conversion funnel: converts
each lane through the scalar [`cvt`](@ref) funnel (so per-lane
specializations and device overrides still apply), then packs. Vectorized
`cvt` specializations call this when their fast path does not cover the
requested combination.
"""
@inline cvt_lanes(::Type{NVector{T,N}}, xs::NTuple{N,Any},
                  mode::RoundingMode, policy::OverflowPolicy) where {T<:Microfloat,N} =
    NVector{T,N}(ntuple(i -> cvt(T, xs[i], mode, policy), Val(N)))

"""
    cvt(::Type{NVector{T,N}}, xs::NTuple{N,Any}, mode, policy) -> NVector{T,N}

Vector form of the conversion funnel: convert `N` source lanes into a packed
`BitPacking.NVector` in one call. The default is lanewise
([`cvt_lanes`](@ref)); specialize on `(T, N, lane type, mode, policy)` for
multi-lane hardware conversions (e.g. PTX `cvt` x2 instructions in device
overlays) or SIMD bit-twiddling over packed sources.

Source containers (`SVector`, `NVector`, any `StaticArray` vector) normalize
to `NTuple` first; packed→packed specializations may intercept the
`NVector`-source signature before it is unpacked.
"""
@inline cvt(::Type{NVector{T,N}}, xs::NTuple{N,Any},
            mode::RoundingMode, policy::OverflowPolicy) where {T<:Microfloat,N} =
    cvt_lanes(NVector{T,N}, xs, mode, policy)
@inline cvt(::Type{NVector{T,N}}, xs::StaticArray{Tuple{N},<:Any,1},
            mode::RoundingMode, policy::OverflowPolicy) where {T<:Microfloat,N} =
    cvt(NVector{T,N}, Tuple(xs), mode, policy)

# Entry points: like the scalar constructors, these only resolve defaults and
# hand off to the funnel. `StaticArray{Tuple{N},<:Real,1}` covers both
# `SVector` and packed `NVector` sources.
(::Type{NVector{T,N}})(xs::StaticArray{Tuple{N},<:Real,1}, mode::RoundingMode;
                       overflow::OverflowPolicy = overflow_policy(T)) where {T<:Microfloat,N} =
    cvt(NVector{T,N}, xs, mode, overflow)
(::Type{NVector{T,N}})(xs::StaticArray{Tuple{N},<:Real,1};
                       overflow::OverflowPolicy = overflow_policy(T)) where {T<:Microfloat,N} =
    cvt(NVector{T,N}, xs, RoundNearest, overflow)
# Same-eltype repacking involves no rounding; this also disambiguates against
# BitPacking's exact-eltype StaticArray constructor.
(::Type{NVector{T,N}})(xs::StaticArray{Tuple{N},T,1}) where {T<:Microfloat,N} =
    NVector{T,N}(Tuple(xs))
