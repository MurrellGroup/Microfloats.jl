import Base: signbit, exponent

"""
    Microfloat <: AbstractFloat

Abstract supertype for byte-sized floating-point numbers. Concrete subtypes
are 8-bit primitive types declared via [`@microfloat`](@ref).

# Examples
```jldoctest
julia> Float8_E4M3 <: Microfloat
true

julia> Float8_E4M3(1.0) + Float8_E4M3(0.5)
Float8_E4M3(1.5)
```
"""
abstract type Microfloat <: AbstractFloat end

sign_mask(::Type{T}) where T<:Microfloat = UInt8((0x01 << sign_bits(T) - 0x01) << (exponent_bits(T) + significand_bits(T)))
exponent_mask(::Type{T}) where T<:Microfloat = UInt8((0x01 << exponent_bits(T) - 0x01) << significand_bits(T))
significand_mask(::Type{T}) where T<:Microfloat = UInt8(0x01 << significand_bits(T) - 0x01)

Base.reinterpret(::Type{Unsigned}, x::Microfloat) = reinterpret(UInt8, x)

signbit(x::Microfloat) = sign_bits(typeof(x)) > 0 && !iszero(reinterpret(Unsigned, x) & sign_mask(typeof(x)))
function exponent(x::T) where T<:Microfloat
    (isnan(x) || isinf(x)) && throw(DomainError(x, "Cannot be NaN or Inf."))
    iszero(x) && throw(DomainError(x, "Cannot be ±0.0."))
    raw = reinterpret(Unsigned, x)
    biased = Int((raw & exponent_mask(T)) >> significand_bits(T))
    biased == 0 || return biased - exponent_bias(T)
    sig = raw & significand_mask(T)
    return 8 - leading_zeros(sig) - exponent_bias(T) - significand_bits(T)
end

function Base.show(io::IO, x::T) where T<:Microfloat
    show_typeinfo = get(IOContext(io), :typeinfo, nothing) != T
    show_typeinfo && print(io, repr(T), "(")
    print(io, Float64(x))
    show_typeinfo && print(io, ")")
    return nothing
end

abstract type NonFiniteBehavior end

"""
    IEEE <: NonFiniteBehavior

IEEE-754-style sentinels: Inf is `exp=all-ones, significand=0`;
NaN is `exp=all-ones, significand≠0`.
"""
abstract type IEEE           <: NonFiniteBehavior end

"""
    NanOnlyAllOnes <: NonFiniteBehavior

NaN is the unique all-ones bit pattern in exponent+significand
(per sign); no Inf encoding. The slot that would otherwise be Inf is
reclaimed for a finite value, extending dynamic range by one step.
"""
abstract type NanOnlyAllOnes <: NonFiniteBehavior end

"""
    FiniteOnly <: NonFiniteBehavior

No Inf or NaN — every bit pattern is a finite value. Requires
`overflow=`[`SAT`](@ref) since no sentinel encoding exists.
"""
abstract type FiniteOnly     <: NonFiniteBehavior end

hasinf(::Type{IEEE})           = true
hasinf(::Type{NanOnlyAllOnes}) = false
hasinf(::Type{FiniteOnly})     = false

hasnan(::Type{IEEE})           = true
hasnan(::Type{NanOnlyAllOnes}) = true
hasnan(::Type{FiniteOnly})     = false

"""
    non_finite_behavior(::Type{<:Microfloat}) -> Type{<:NonFiniteBehavior}

Return [`IEEE`](@ref), [`NanOnlyAllOnes`](@ref), or [`FiniteOnly`](@ref)
based on the trait registered for the concrete type by [`@microfloat`](@ref).

# Examples
```jldoctest
julia> Microfloats.non_finite_behavior(Float8_E5M2)
Microfloats.IEEE

julia> Microfloats.non_finite_behavior(Float8_E4M3FN)
Microfloats.NanOnlyAllOnes

julia> Microfloats.non_finite_behavior(Float4_E2M1FN)
Microfloats.FiniteOnly
```
"""
non_finite_behavior(::Type{T}) where T<:Microfloat =
    error("$T must define `Microfloats.non_finite_behavior(::Type{$T})`")

"""
    hasinf(::Type{<:Microfloat}) -> Bool

Return `true` if the type can represent Inf, otherwise `false`.

# Examples
```jldoctest
julia> Microfloats.hasinf(Float8_E5M2)
true

julia> Microfloats.hasinf(Float8_E4M3FN)
false
```
"""
hasinf(::Type{T}) where T<:Microfloat = hasinf(non_finite_behavior(T))

"""
    hasnan(::Type{<:Microfloat}) -> Bool

Return `true` if the type can represent NaN, otherwise `false`.

# Examples
```jldoctest
julia> Microfloats.hasnan(Float8_E4M3FN)
true

julia> Microfloats.hasnan(Float4_E2M1FN)
false
```
"""
hasnan(::Type{T}) where T<:Microfloat = hasnan(non_finite_behavior(T))

# ───────────────────────── Inf / NaN / floatmax / inf / nan ──────────────────────────

Base.isinf(x::T) where T<:Microfloat = _isinf(non_finite_behavior(T), x)
Base.isnan(x::T) where T<:Microfloat = _isnan(non_finite_behavior(T), x)

inf(::Type{T}) where T<:Microfloat = _inf(non_finite_behavior(T), T)
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

Base.typemin(::Type{T}) where T<:Microfloat =
    sign_bits(T) == 0 ? zero(T) : hasinf(T) ? -inf(T) : -floatmax(T)
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
Base.:(-)(x::T) where T<:Microfloat =
    sign_bits(T) == 0 ? throw(DomainError(x, "cannot negate unsigned $T")) :
    reinterpret(T, sign_mask(T) ⊻ reinterpret(Unsigned, x))
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
