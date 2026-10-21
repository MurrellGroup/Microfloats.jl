import Base: signbit, exponent

"""
    Microfloat <: AbstractFloat

Abstract supertype for byte-sized floating-point numbers. Concrete subtypes
are primitive types declared via [`@microfloat`](@ref), one byte each, with
the value in the low [`bitwidth`](@ref Microfloats.bitwidth) bits.

On Julia 1.14 and later, a type narrower than 8 bits is declared with its
real width (`Core.bitsizeof(Float4_E2M1FN) == 4`). Arrays of it can then not
be reinterpreted, but `reinterpret` between a single value and `UInt8` still
works, dropping the bits above the width.

Before 1.14 every microfloat is an 8-bit type. Operations ignore the bits
above the width, which `reinterpret` can set, and never set them themselves.
`===` alone still compares the whole byte.

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

value_mask(::Type{T}) where T<:Microfloat = 0xff >> (8 - bitwidth(T))

@static if isdefined(Core, :bitsizeof)
    # Julia 1.14 lets a primitive type declare fewer bits than it stores, and
    # `@microfloat` declares a type narrower than a byte with its real width.
    declared_bits(N::Int) = N
    has_unused_bits(::Type{T}) where T<:Microfloat = Core.bitsizeof(T) < 8
else
    declared_bits(N::Int) = 8
    has_unused_bits(::Type{<:Microfloat}) = false
end

# A microfloat's value is the low `bitwidth(T)` bits of its byte. A type that
# declares only those bits is widened and truncated. In an 8-bit type the bits
# above can be set by `reinterpret(T, ::UInt8)`, so every operation reads
# values through `raw_bits`, which ignores them, and builds values through
# `from_raw_bits` from bits that fit the width. Both use intrinsics, because
# `reinterpret` is defined through them on Julia 1.14.
@inline function raw_bits(x::T) where T<:Microfloat
    has_unused_bits(T) && return Core.Intrinsics.zext_int(UInt8, x)
    return Core.Intrinsics.bitcast(UInt8, x) & value_mask(T)
end
@inline function from_raw_bits(::Type{T}, bits::UInt8) where T<:Microfloat
    has_unused_bits(T) && return Core.Intrinsics.trunc_int(T, bits)
    return Core.Intrinsics.bitcast(T, bits)
end

Base.reinterpret(::Type{Unsigned}, x::Microfloat) = raw_bits(x)

@static if isdefined(Core, :bitsizeof)
    # `reinterpret` refuses a type narrower than its storage. A microfloat's
    # byte holds its value in the low bits, so these keep the conversions to
    # and from that byte working. Arrays of such types still cannot be
    # reinterpreted.
    Base.reinterpret(::Type{UInt8}, x::Microfloat) = raw_bits(x)
    Base.reinterpret(::Type{T}, bits::UInt8) where T<:Microfloat = from_raw_bits(T, bits)
end

signbit(x::Microfloat) = sign_bits(typeof(x)) > 0 && !iszero(raw_bits(x) & sign_mask(typeof(x)))
function exponent(x::T) where T<:Microfloat
    (isnan(x) || isinf(x)) && throw(DomainError(x, "Cannot be NaN or Inf."))
    iszero(x) && throw(DomainError(x, "Cannot be ±0.0."))
    raw = raw_bits(x)
    biased = Int((raw & exponent_mask(T)) >> significand_bits(T))
    biased == 0 || return biased - exponent_bias(T)
    sig = raw & significand_mask(T)
    return 8 - leading_zeros(sig) - exponent_bias(T) - significand_bits(T)
end

function Base.show(io::IO, x::T) where T<:Microfloat
    show_typeinfo = get(IOContext(io), :typeinfo, nothing) != T
    show_typeinfo && print(io, repr(T), "(")
    print(io, decimal_string(x))
    show_typeinfo && print(io, ")")
    return nothing
end

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
    raw = raw_bits(x)
    (raw & exponent_mask(T)) == exponent_mask(T) && iszero(raw & significand_mask(T))
end
function _isnan(::Type{IEEE}, x::T) where T<:Microfloat
    raw = raw_bits(x)
    (raw & exponent_mask(T)) == exponent_mask(T) && !iszero(raw & significand_mask(T))
end
_inf(::Type{IEEE}, ::Type{T}) where T<:Microfloat = from_raw_bits(T, exponent_mask(T))
_nan(::Type{IEEE}, ::Type{T}) where T<:Microfloat =
    from_raw_bits(T, exponent_mask(T) | (UInt8(0x01) << (significand_bits(T) - 1)))
_floatmax(::Type{IEEE}, ::Type{T}) where T<:Microfloat =
    from_raw_bits(T, (exponent_mask(T) - (UInt8(0x01) << significand_bits(T))) | significand_mask(T))

# NanOnlyAllOnes
_isinf(::Type{NanOnlyAllOnes}, ::Microfloat) = false
function _isnan(::Type{NanOnlyAllOnes}, x::T) where T<:Microfloat
    raw = raw_bits(x)
    (raw & ~sign_mask(T)) == (exponent_mask(T) | significand_mask(T))
end
_inf(::Type{NanOnlyAllOnes}, ::Type{T}) where T<:Microfloat =
    throw(DomainError(T, "$T has no Inf"))
_nan(::Type{NanOnlyAllOnes}, ::Type{T}) where T<:Microfloat =
    from_raw_bits(T, exponent_mask(T) | significand_mask(T))
_floatmax(::Type{NanOnlyAllOnes}, ::Type{T}) where T<:Microfloat =
    from_raw_bits(T, (exponent_mask(T) | significand_mask(T)) - UInt8(0x01))

# FiniteOnly
_isinf(::Type{FiniteOnly}, ::Microfloat) = false
_isnan(::Type{FiniteOnly}, ::Microfloat) = false
_inf(::Type{FiniteOnly}, ::Type{T}) where T<:Microfloat =
    throw(DomainError(T, "$T has no Inf"))
_nan(::Type{FiniteOnly}, ::Type{T}) where T<:Microfloat =
    throw(DomainError(T, "$T has no NaN"))
_floatmax(::Type{FiniteOnly}, ::Type{T}) where T<:Microfloat =
    from_raw_bits(T, exponent_mask(T) | significand_mask(T))

# ───────────────────────── generic Base methods ──────────────────────────

Base.typemin(::Type{T}) where T<:Microfloat =
    sign_bits(T) == 0 ? zero(T) : hasinf(T) ? -inf(T) : -floatmax(T)
Base.typemax(::Type{T}) where T<:Microfloat = hasinf(T) ? inf(T) : floatmax(T)

function Base.floatmin(::Type{T}) where T<:Microfloat
    significand_bits(T) == 0 && return from_raw_bits(T, 0x00)
    bits = UInt8(0x01) << significand_bits(T)
    hasinf(T) && bits == raw_bits(inf(T)) &&
        throw(DomainError(T, "$T has no normal values"))
    return from_raw_bits(T, bits)
end

Base.zero(::Type{T}) where T<:Microfloat = from_raw_bits(T, 0x00)
Base.one(::Type{T}) where T<:Microfloat = T(true)

Base.eps(x::Microfloat) = max(x - prevfloat(x), nextfloat(x) - x)
Base.eps(T::Type{<:Microfloat}) = eps(one(T))

Base.abs(x::T) where T<:Microfloat = from_raw_bits(T, raw_bits(x) & ~sign_mask(T))
Base.iszero(x::T) where T<:Microfloat = significand_bits(T) == 0 ? false : abs(x) === zero(T)
Base.:(-)(x::T) where T<:Microfloat =
    sign_bits(T) == 0 ? throw_negate_unsigned(T, x) :
    from_raw_bits(T, sign_mask(T) ⊻ raw_bits(x))
Base.Bool(x::T) where T<:Microfloat = iszero(x) ? false : isone(x) ? true : throw(InexactError(:Bool, Bool, x))

Base.precision(::Type{T}) where T<:Microfloat = significand_bits(T) + 1

Base.sign(x::Microfloat) = ifelse(isnan(x) | iszero(x), x, ifelse(signbit(x), -one(x), one(x)))

Base.round(x::T, r::RoundingMode; kws...) where T<:Microfloat = T(round(Float32(x), r; kws...))
# Base defines these modes on `AbstractFloat`, which is more specific in the
# mode and less specific in the value than the method above.
for M in (:FromZero, :NearestTiesAway, :NearestTiesUp)
    @eval Base.round(x::T, r::RoundingMode{$(QuoteNode(M))}) where T<:Microfloat = T(round(Float32(x), r))
end

Base.issubnormal(x::T) where T<:Microfloat =
    0x00 < (raw_bits(x) & ~sign_mask(T)) <= (UInt8(0x01) << significand_bits(T)) - 0x01

ispositive(x::T) where T<:Microfloat = iszero(raw_bits(x) & sign_mask(T))

function Base.nextfloat(x::T) where T<:Microfloat
    if isnan(x)
        return x
    elseif isinf(x)
        return ispositive(x) ? inf(T) : -floatmax(T)
    elseif iszero(x)
        return from_raw_bits(T, 0x01)
    elseif ispositive(x)
        # Without Inf, floatmax is as far as it goes, like Inf in Base.
        !hasinf(T) && raw_bits(x) == raw_bits(floatmax(T)) && return floatmax(T)
        return from_raw_bits(T, raw_bits(x) + 0x01)
    else
        return from_raw_bits(T, raw_bits(x) - 0x01)
    end
end

function Base.prevfloat(x::T) where T<:Microfloat
    if isnan(x)
        return x
    elseif isinf(x)
        return ispositive(x) ? floatmax(T) : -inf(T)
    elseif iszero(x)
        sign_bits(T) == 0 && return x
        return from_raw_bits(T, sign_mask(T) | 0x01)
    elseif ispositive(x)
        raw = raw_bits(x)
        raw == 0x00 && sign_bits(T) == 0 && return x
        return from_raw_bits(T, raw - 0x01)
    else
        !hasinf(T) && raw_bits(x) == raw_bits(-floatmax(T)) && return -floatmax(T)
        return from_raw_bits(T, raw_bits(x) + 0x01)
    end
end

Base.decompose(x::T) where T<:Microfloat = Base.decompose(Float32(x))

Base.widen(::Type{T}) where T<:Microfloat = BFloat16

Base.promote_rule(::Type{M}, ::Type{T}) where {M<:Microfloat,T<:Union{BFloat16,Float16,Float32,Float64}} = T
Base.promote_rule(::Type{M}, ::Type{T}) where {M<:Microfloat,T<:Integer} = M
Base.promote_rule(::Type{M}, ::Type{M}) where {M<:Microfloat} = M
