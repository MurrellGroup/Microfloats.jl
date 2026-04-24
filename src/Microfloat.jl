@republic import Base: signbit, exponent

"""
    Microfloat{S,E,M} <: AbstractFloat

Abstract type for within-byte floating-point numbers with `S` sign bits
(0 or 1), `E` exponent bits (≥ 1), and `M` significand bits (≥ 0),
with `S + E + M ≤ 8`.

Concrete subtypes are 8-bit `primitive type`s that must also register
a [`non_finite_behavior`](@ref). See [`@microfloat`](@ref) for the
macro-based convenience declaration.
"""
abstract type Microfloat{S,E,M} <: AbstractFloat end

float_bits(::Type{<:Microfloat{S,E,M}}) where {S,E,M} = (S,E,M)

sign_mask(::Type{T}) where T<:Microfloat = UInt8((0x01 << sign_bits(T) - 0x01) << (exponent_bits(T) + significand_bits(T)))
exponent_mask(::Type{T}) where T<:Microfloat = UInt8((0x01 << exponent_bits(T) - 0x01) << significand_bits(T))
significand_mask(::Type{T}) where T<:Microfloat = UInt8(0x01 << significand_bits(T) - 0x01)

Base.reinterpret(::Type{Unsigned}, x::Microfloat) = reinterpret(UInt8, x)

signbit(x::Microfloat) = sign_bits(typeof(x)) > 0 && !iszero(reinterpret(Unsigned, x) & sign_mask(typeof(x)))
exponent(x::Microfloat) = Int((reinterpret(Unsigned, x) & exponent_mask(typeof(x))) >> significand_bits(typeof(x)))

function Base.show(io::IO, x::T) where T<:Microfloat
    show_typeinfo = get(IOContext(io), :typeinfo, nothing) != T
    show_typeinfo && print(io, repr(T), "(")
    print(io, Float64(x))
    show_typeinfo && print(io, ")")
    return nothing
end

"""
    NonFiniteBehavior

Trait hierarchy describing how a [`Microfloat`](@ref) type encodes non-finite
values. Each concrete `Microfloat` subtype registers its behavior by defining
a [`non_finite_behavior`](@ref) method.

Three behaviors:

- [`IEEE`](@ref): exponent all-ones with zero significand ⇒ Inf;
  all-ones exponent with nonzero significand ⇒ NaN.
- [`NanOnlyAllOnes`](@ref): no Inf. The single NaN encoding has all
  exponent and significand bits set.
- [`FiniteOnly`](@ref): no Inf and no NaN — every bit pattern is finite.
  Matches MX sub-byte types and `F4E2M1FN`.
"""
abstract type NonFiniteBehavior end

"""IEEE-754-style encoding of Inf and NaN. Requires `M ≥ 1`."""
abstract type IEEE           <: NonFiniteBehavior end

"""NaN encoded as all-ones in exponent+significand; no Inf."""
abstract type NanOnlyAllOnes <: NonFiniteBehavior end

"""No Inf or NaN — every bit pattern is a finite value."""
abstract type FiniteOnly     <: NonFiniteBehavior end

hasinf(::Type{IEEE})           = true
hasinf(::Type{NanOnlyAllOnes}) = false
hasinf(::Type{FiniteOnly})     = false

hasnan(::Type{IEEE})           = true
hasnan(::Type{NanOnlyAllOnes}) = true
hasnan(::Type{FiniteOnly})     = false

"""
    non_finite_behavior(T) -> Type{<:NonFiniteBehavior}

Required trait method on every concrete [`Microfloat`](@ref) subtype.
Returns one of `IEEE`, `NanOnlyAllOnes`, or `FiniteOnly`.
"""
non_finite_behavior(::Type{T}) where T<:Microfloat =
    error("$T must define `Microfloats.non_finite_behavior(::Type{$T})`")

hasinf(::Type{T}) where T<:Microfloat = hasinf(non_finite_behavior(T))
hasnan(::Type{T}) where T<:Microfloat = hasnan(non_finite_behavior(T))

# ───────────────────────── Inf / NaN / floatmax / inf / nan ──────────────────────────

Base.isinf(x::T) where T<:Microfloat = _isinf(non_finite_behavior(T), x)
Base.isnan(x::T) where T<:Microfloat = _isnan(non_finite_behavior(T), x)

"""Bit pattern for +Inf. Throws if the type has no Inf."""
inf(::Type{T}) where T<:Microfloat = _inf(non_finite_behavior(T), T)

"""Bit pattern for NaN. Throws if the type has no NaN."""
nan(::Type{T}) where T<:Microfloat = _nan(non_finite_behavior(T), T)

Base.floatmax(::Type{T}) where T<:Microfloat = _floatmax(non_finite_behavior(T), T)

# IEEE
function _isinf(::Type{IEEE}, x::T) where T<:Microfloat
    raw = reinterpret(Unsigned, x)
    (raw & exponent_mask(T)) == exponent_mask(T) && iszero(raw & significand_mask(T))
end
function _isnan(::Type{IEEE}, x::T) where T<:Microfloat
    raw = reinterpret(Unsigned, x)
    (raw & exponent_mask(T)) == exponent_mask(T) && !iszero(raw & significand_mask(T))
end
_inf(::Type{IEEE}, ::Type{T}) where T<:Microfloat = reinterpret(T, exponent_mask(T))
_nan(::Type{IEEE}, ::Type{T}) where T<:Microfloat =
    reinterpret(T, exponent_mask(T) | (UInt8(0x01) << (significand_bits(T) - 1)))
_floatmax(::Type{IEEE}, ::Type{T}) where T<:Microfloat =
    reinterpret(T, (exponent_mask(T) - (UInt8(0x01) << significand_bits(T))) | significand_mask(T))

# NanOnlyAllOnes
_isinf(::Type{NanOnlyAllOnes}, ::Microfloat) = false
function _isnan(::Type{NanOnlyAllOnes}, x::T) where T<:Microfloat
    raw = reinterpret(Unsigned, x)
    (raw & ~sign_mask(T)) == (exponent_mask(T) | significand_mask(T))
end
_inf(::Type{NanOnlyAllOnes}, ::Type{T}) where T<:Microfloat =
    throw(DomainError(T, "$T has no Inf"))
_nan(::Type{NanOnlyAllOnes}, ::Type{T}) where T<:Microfloat =
    reinterpret(T, exponent_mask(T) | significand_mask(T))
_floatmax(::Type{NanOnlyAllOnes}, ::Type{T}) where T<:Microfloat =
    reinterpret(T, (exponent_mask(T) | significand_mask(T)) - UInt8(0x01))

# FiniteOnly
_isinf(::Type{FiniteOnly}, ::Microfloat) = false
_isnan(::Type{FiniteOnly}, ::Microfloat) = false
_inf(::Type{FiniteOnly}, ::Type{T}) where T<:Microfloat =
    throw(DomainError(T, "$T has no Inf"))
_nan(::Type{FiniteOnly}, ::Type{T}) where T<:Microfloat =
    throw(DomainError(T, "$T has no NaN"))
_floatmax(::Type{FiniteOnly}, ::Type{T}) where T<:Microfloat =
    reinterpret(T, exponent_mask(T) | significand_mask(T))

# ───────────────────────── generic Base methods ──────────────────────────

Base.typemin(::Type{T}) where T<:Microfloat{0} = zero(T)
Base.typemin(::Type{T}) where T<:Microfloat = hasinf(T) ? -inf(T) : -floatmax(T)
Base.typemax(::Type{T}) where T<:Microfloat = hasinf(T) ? inf(T) : floatmax(T)

Base.floatmin(::Type{T}) where T<:Microfloat =
    significand_bits(T) == 0 ? reinterpret(T, 0x00) :
    reinterpret(T, UInt8(0x01) << significand_bits(T))

Base.zero(::Type{T}) where T<:Microfloat = reinterpret(T, 0x00)
Base.one(::Type{T}) where T<:Microfloat = T(true)

Base.eps(x::Microfloat) = max(x - prevfloat(x), nextfloat(x) - x)
Base.eps(T::Type{<:Microfloat}) = eps(one(T))

Base.abs(x::T) where T<:Microfloat = reinterpret(T, reinterpret(Unsigned, x) & ~sign_mask(T))
Base.iszero(x::T) where T<:Microfloat = significand_bits(T) == 0 ? false : abs(x) === zero(T)
Base.:(-)(x::T) where T<:Microfloat{0} = throw(DomainError(x, "cannot negate unsigned $T"))
Base.:(-)(x::T) where T<:Microfloat = reinterpret(T, sign_mask(T) ⊻ reinterpret(Unsigned, x))
Base.Bool(x::T) where T<:Microfloat = iszero(x) ? false : isone(x) ? true : throw(InexactError(:Bool, Bool, x))

Base.precision(::Type{T}) where T<:Microfloat = significand_bits(T) + 1

Base.sign(x::Microfloat) = ifelse(isnan(x) | iszero(x), x, ifelse(signbit(x), -one(x), one(x)))

Base.round(x::T, r::RoundingMode; kws...) where T<:Microfloat = T(round(Float32(x), r; kws...))

Base.issubnormal(x::T) where T<:Microfloat =
    0x00 < (reinterpret(Unsigned, x) & ~sign_mask(T)) <= (UInt8(0x01) << significand_bits(T)) - 0x01

ispositive(x::T) where T<:Microfloat = iszero(reinterpret(Unsigned, x) & sign_mask(T))

function Base.nextfloat(x::T) where T<:Microfloat
    if isnan(x)
        return x
    elseif isinf(x)
        return ispositive(x) ? inf(T) : -floatmax(T)
    elseif iszero(x)
        return reinterpret(T, 0x01)
    elseif ispositive(x)
        return reinterpret(T, reinterpret(Unsigned, x) + 0x01)
    else
        return reinterpret(T, reinterpret(Unsigned, x) - 0x01)
    end
end

function Base.prevfloat(x::T) where T<:Microfloat
    if isnan(x)
        return x
    elseif isinf(x)
        return ispositive(x) ? floatmax(T) : -inf(T)
    elseif iszero(x)
        sign_bits(T) == 0 && return x
        return reinterpret(T, sign_mask(T) | 0x01)
    elseif ispositive(x)
        raw = reinterpret(Unsigned, x)
        raw == 0x00 && sign_bits(T) == 0 && return x
        return reinterpret(T, raw - 0x01)
    else
        return reinterpret(T, reinterpret(Unsigned, x) + 0x01)
    end
end

Base.decompose(x::T) where T<:Microfloat = Base.decompose(BFloat16(x))

Base.widen(::Type{T}) where T<:Microfloat = BFloat16

Base.promote_rule(::Type{M}, ::Type{T}) where {M<:Microfloat,T<:Union{BFloat16,Float16,Float32,Float64}} = T
Base.promote_rule(::Type{M}, ::Type{T}) where {M<:Microfloat,T<:Integer} = M
Base.promote_rule(::Type{M}, ::Type{M}) where {M<:Microfloat} = M
