abstract type Variant end

primitive type Microfloat{S,E,M,V<:Variant} <: AbstractFloat 8 end

abstract type IEEE <: Variant end

const IEEEMicrofloat{S,E,M} = Microfloat{S,E,M,IEEE}

"""
    Float8_E3M4

8-bit Microfloat with 1 sign bit, 3 exponent bits, and 4 mantissa bits.
Follows the pattern of the IEEE 754 standard.
"""
const Float8_E3M4 = IEEEMicrofloat{1,3,4}

const Float8_E4M3 = IEEEMicrofloat{1,4,3}
const Float8_E5M2 = IEEEMicrofloat{1,5,2}
const Float6_E2M3 = IEEEMicrofloat{1,2,3}
const Float6_E3M2 = IEEEMicrofloat{1,3,2}
const Float4_E2M1 = IEEEMicrofloat{1,2,1}

"""
    Microfloat(S, E, M, V=IEEE)

Create a new `Microfloat` type with `S` sign bits, `E` exponent bits, and `M` mantissa bits.

This "type constructor" ensures that the resulting type is legal.

The `V` argument can be set to `:MX` to create a Microscaling Format (MX) type.
"""
function Microfloat(S::Int, E::Int, M::Int, V::Type{<:Variant}=IEEE)
    S in (0, 1) || throw(ArgumentError("sign bit must be 0 or 1"))
    E >= 1 || throw(ArgumentError("number of exponent bits must be non-negative"))
    M >= 0 || throw(ArgumentError("number of mantissa bits must be non-negative"))
    0 < S + E + M <= 8 || throw(ArgumentError("total number of bits must be between 1 and 8"))
    return Microfloat{S,E,M,V}
end

Microfloat{S}(E::Int, M::Int; kws...) where S = Microfloat(S, E, M; kws...)

uint(::Type{T}) where T<:Microfloat = UInt8
n_sign_bits(::Type{T}) where {S,T<:Microfloat{S}} = S
n_exponent_bits(::Type{T}) where {E,T<:Microfloat{<:Any,E}} = E
n_mantissa_bits(::Type{T}) where {M,T<:Microfloat{<:Any,<:Any,M}} = M

Base.hash(x::Microfloat, h::UInt) = hash(Float32(x), h)

inf(::Type{T}) where T<:Microfloat = reinterpret(T, exponent_mask(T))
nan(::Type{T}) where T<:Microfloat = has_mantissa(T) ?
    reinterpret(T, exponent_mask(T) | (bit_ones(1, uint(T)) << mantissa_offset(T))) :
    throw(DomainError(T, "$T has no NaN values"))

Base.isinf(x::T) where T<:Microfloat = x === inf(T) || x === -inf(T)
Base.isnan(x::T) where T<:Microfloat = only_exponent(x) === exponent_mask(T) && !iszero(only_mantissa(x))

Base.zero(::Type{T}) where T<:Microfloat = reinterpret(T, 0x00)
Base.one(::Type{T}) where T<:Microfloat = T(1.0f0)

Base.eps(x::Microfloat) = max(x-prevfloat(x), nextfloat(x)-x)
Base.eps(T::Type{<:Microfloat}) = eps(one(T))

Base.floatmin(::Type{T}) where T<:Microfloat = n_exponent_bits(T) > 1 ? reinterpret(T, bit_ones(1) << exponent_offset(T)) : throw(DomainError(T, "$T has no normal numbers"))
Base.floatmax(::Type{T}) where T<:Microfloat = reinterpret(T, bit_ones(n_exponent_bits(T) - 1) << (exponent_offset(T) + 1) | mantissa_mask(T))

Base.typemin(::Type{T}) where T<:Microfloat = -inf(T)
Base.typemin(::Type{T}) where T<:Microfloat{0} = zero(T)

Base.typemax(::Type{T}) where T<:Microfloat = inf(T)

Base.abs(x::T) where T<:Microfloat = reinterpret(T, reinterpret(UInt8, x) & ~sign_mask(T))
Base.iszero(x::T) where T<:Microfloat = abs(x) === zero(T)
Base.:(-)(x::T) where T<:Microfloat = reinterpret(T, sign_mask(T) ⊻ reinterpret(UInt8, x))
Base.Bool(x::T) where T<:Microfloat = iszero(x) ? false : isone(x) ? true : throw(InexactError(:Bool, Bool, x))

Base.precision(::Type{T}) where T<:Microfloat = n_mantissa_bits(T) + 1

Base.signbit(x::Microfloat) = x !== abs(x)
Base.sign(x::Microfloat) = ifelse(isnan(x) || iszero(x), x, ifelse(signbit(x), -one(x), one(x)))

Base.round(x::T, r::RoundingMode; kws...) where T<:Microfloat = T(round(Float32(x), r; kws...))

function Base.nextfloat(x::T) where T<:Microfloat
    if isnan(x)
        return x
    elseif isinf(x)
        return x
    elseif x === -inf(T)
        return -floatmax(T)
    elseif iszero(x)
        return reinterpret(T, bit_ones(1) << mantissa_offset(T))
    elseif ispositive(x)
        return reinterpret(T, reinterpret(UInt8, x) + bit_ones(1) << mantissa_offset(T))
    else
        return reinterpret(T, reinterpret(UInt8, x) - bit_ones(1) << mantissa_offset(T))
    end
end

function Base.prevfloat(x::T) where T<:Microfloat
    if isnan(x)
        return x
    elseif x === inf(T)
        return floatmax(T)
    elseif x === -inf(T) # this needs to be after +Inf check for unsigned types to work
        return x
    elseif iszero(x)
        return reinterpret(T, sign_mask(T) | (bit_ones(1) << mantissa_offset(T)))
    elseif ispositive(x)
        return reinterpret(T, reinterpret(UInt8, x) - bit_ones(1) << mantissa_offset(T))
    else
        return reinterpret(T, reinterpret(UInt8, x) + bit_ones(1) << mantissa_offset(T))
    end
end

const STANDARD_FLOATS = Union{Float16, Float32, Float64, BigFloat}

# TODO: check if these rules are even doing anything

Base.promote_rule(::Type{<:Microfloat}, ::Type{T}) where T<:STANDARD_FLOATS = Float32
Base.promote_rule(::Type{<:Microfloat}, ::Type{T}) where T<:Integer = Float32

# Microfloat vs Microfloat: dominance within same variant, else Float32
Base.promote_rule(::Type{Microfloat{S1,E1,M1,V}}, ::Type{Microfloat{S2,E2,M2,V}}) where {S1,E1,M1,S2,E2,M2,V} =
    (S1 ≤ S2 && E1 ≤ E2 && M1 ≤ M2) ? Microfloat{S2,E2,M2,V} :
    (S2 ≤ S1 && E2 ≤ E1 && M2 ≤ M1) ? Microfloat{S1,E1,M1,V} :
    Float32

# Cross-variant: fall back to Float32
Base.promote_rule(::Type{<:Microfloat}, ::Type{<:Microfloat}) = Float32
