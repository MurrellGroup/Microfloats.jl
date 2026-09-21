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
#
# NVIDIA's fp6x2/fp6x4 give each value its own byte, in the low six bits:
# the layout PTX `cvt` produces and the one `mma` reads for `.kind::f8f6f4`.
# A Microfloat scalar is already one byte with its value in the low bits, so
# that layout is an `SVector`. Densely packed 6-bit storage, as in memory
# and TMA transfers, is `NVector{Float6_E2M3FN,N}`.

const Float6x2_E2M3FN = SVector{2,Float6_E2M3FN}
const Float6x2_E3M2FN = SVector{2,Float6_E3M2FN}

const Float6x4_E2M3FN = SVector{4,Float6_E2M3FN}
const Float6x4_E3M2FN = SVector{4,Float6_E3M2FN}

# 4-bit

const Float4x2_E2M1FN = NVector{Float4_E2M1FN,2}
const Float4x4_E2M1FN = NVector{Float4_E2M1FN,4}

# ───────────────────────── vector conversion funnel ──────────────────────────
#
# Two vector destinations share one funnel. `SVector{N,T}` holds one lane per
# byte; `NVector{T,N}` packs the same lanes densely. Packing is layout only,
# so the packed default converts through the unpacked funnel and then packs:
# a multi-lane specialization of the `SVector` form, such as a native PTX
# pair conversion, therefore also serves dense destinations.

const LaneVector{T,N} = Union{SVector{N,T},NVector{T,N}}

"""
    cvt_lanes(::Type{SVector{N,T}}, xs::NTuple{N,Any}, mode, policy) -> SVector{N,T}
    cvt_lanes(::Type{NVector{T,N}}, xs::NTuple{N,Any}, mode, policy) -> NVector{T,N}

Reference lanewise implementation of the vector conversion funnel: converts
each lane through the scalar [`cvt`](@ref) funnel (so per-lane
specializations and device overrides still apply), then builds the vector.
Vectorized `cvt` specializations call this when their fast path does not
cover the requested combination.
"""
@inline cvt_lanes(::Type{SVector{N,T}}, xs::NTuple{N,Any},
                  mode::RoundingMode, policy::OverflowPolicy) where {T<:Microfloat,N} =
    SVector{N,T}(ntuple(i -> cvt(T, xs[i], mode, policy), Val(N)))
@inline cvt_lanes(::Type{NVector{T,N}}, xs::NTuple{N,Any},
                  mode::RoundingMode, policy::OverflowPolicy) where {T<:Microfloat,N} =
    NVector{T,N}(ntuple(i -> cvt(T, xs[i], mode, policy), Val(N)))

"""
    cvt(::Type{SVector{N,T}}, xs::NTuple{N,Any}, mode, policy) -> SVector{N,T}
    cvt(::Type{NVector{T,N}}, xs::NTuple{N,Any}, mode, policy) -> NVector{T,N}

Vector forms of the conversion funnel: convert `N` source lanes in one call,
into one byte per lane (`SVector`) or densely packed (`BitPacking.NVector`).
The `SVector` default is lanewise ([`cvt_lanes`](@ref)); the `NVector`
default converts through the `SVector` form and packs. Packing is layout
only and folds away, so specialize the `SVector` form for multi-lane hardware
conversions (e.g. PTX `cvt` x2 instructions in device overlays): that serves
both destinations. Specialize the `NVector` form only for conversions that
work on dense storage directly, such as bit-twiddling over packed sources.

Source containers (`SVector`, `NVector`, any `StaticArray` vector) normalize
to `NTuple` first; packed→packed specializations may intercept the
`NVector`-source signature before it is unpacked.
"""
@inline cvt(::Type{SVector{N,T}}, xs::NTuple{N,Any},
            mode::RoundingMode, policy::OverflowPolicy) where {T<:Microfloat,N} =
    cvt_lanes(SVector{N,T}, xs, mode, policy)
@inline cvt(::Type{NVector{T,N}}, xs::NTuple{N,Any},
            mode::RoundingMode, policy::OverflowPolicy) where {T<:Microfloat,N} =
    NVector{T,N}(Tuple(cvt(SVector{N,T}, xs, mode, policy)))
@inline cvt(::Type{V}, xs::StaticArray{Tuple{N},<:Any,1},
            mode::RoundingMode, policy::OverflowPolicy) where {T<:Microfloat,N,V<:LaneVector{T,N}} =
    cvt(V, Tuple(xs), mode, policy)

# Entry points: like the scalar constructors, these only resolve defaults and
# hand off to the funnel. `StaticArray{Tuple{N},<:Real,1}` covers both
# `SVector` and packed `NVector` sources.
for V in (:(SVector{N,T}), :(NVector{T,N}))
    @eval begin
        (::Type{$V})(xs::StaticArray{Tuple{N},<:Real,1}, mode::RoundingMode;
                     overflow::OverflowPolicy = overflow_policy(T)) where {T<:Microfloat,N} =
            cvt($V, xs, mode, overflow)
        (::Type{$V})(xs::StaticArray{Tuple{N},<:Real,1};
                     overflow::OverflowPolicy = overflow_policy(T)) where {T<:Microfloat,N} =
            cvt($V, xs, RoundNearest, overflow)
        # Same-eltype relayout involves no rounding; this also disambiguates
        # against the exact-eltype StaticArray constructors.
        (::Type{$V})(xs::StaticArray{Tuple{N},T,1}) where {T<:Microfloat,N} =
            $V(Tuple(xs))
    end
end

# BitPacking unpacks any packed vector into an SArray. For Microfloat
# destinations these two methods take precedence: a different source eltype
# converts through the funnel, the same eltype only unpacks.
(::Type{SVector{N,T}})(xs::NVector{<:Real,N};
                       overflow::OverflowPolicy = overflow_policy(T)) where {T<:Microfloat,N} =
    cvt(SVector{N,T}, xs, RoundNearest, overflow)
(::Type{SVector{N,T}})(xs::NVector{T,N}) where {T<:Microfloat,N} =
    SVector{N,T}(Tuple(xs))

# ───────────────────────── vector widening funnel ──────────────────────────

"""
    cvt(::Type{SVector{N,F}}, xs::NTuple{N,T}) -> SVector{N,F}

Vector form of the widening funnel: `N` lanes of microfloat `T` to `N` wide
lanes (`F` is `Float16`, `BFloat16`, `Float32` or `Float64`). Packed
(`NVector`) and unpacked (`SVector`) sources normalize to a tuple of lanes;
the default is lanewise through the scalar widening funnel. Specialize it for
multi-lane hardware conversions, e.g. PTX `cvt.rn.bf16x2.e4m3x2` in device
overlays. Packing and unpacking a source is layout only and folds away, so a
specialization on the tuple form serves every source container.
"""
@inline cvt(::Type{SVector{N,F}}, xs::NTuple{N,T}) where {N,F<:WideFloat,T<:Microfloat} =
    cvt_lanes(SVector{N,F}, xs)
@inline cvt_lanes(::Type{SVector{N,F}}, xs::NTuple{N,T}) where {N,F<:WideFloat,T<:Microfloat} =
    SVector{N,F}(ntuple(i -> cvt(F, xs[i]), Val(N)))
@inline cvt(::Type{SVector{N,F}}, xs::StaticArray{Tuple{N},T,1}) where {N,F<:WideFloat,T<:Microfloat} =
    cvt(SVector{N,F}, Tuple(xs))

# Entry points. One method per destination eltype, like the scalar
# constructors, so they stay more specific than the generic StaticArrays and
# BitPacking constructors in both arguments.
for F in (Float16, BFloat16, Float32, Float64)
    @eval begin
        (::Type{SVector{N,$F}})(xs::SVector{N,T}) where {N,T<:Microfloat} =
            cvt(SVector{N,$F}, xs)
        (::Type{SVector{N,$F}})(xs::NVector{T,N}) where {N,T<:Microfloat} =
            cvt(SVector{N,$F}, xs)
    end
end
