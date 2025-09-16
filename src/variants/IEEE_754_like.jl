"""
    IEEE_754_like
"""
abstract type IEEE_754_like end

const IEEEFloat{S,E,M} = Microfloat{S,E,M,IEEE_754_like}

hasinf(::Type{<:IEEEFloat}) = true
hasnan(::Type{<:IEEEFloat}) = true

inf(::Type{T}) where T<:IEEEFloat = reinterpret(T, exponent_mask(T))
nan(::Type{T}) where T<:IEEEFloat = reinterpret(T, exponent_mask(T) | 0x01 << (significand_bits(T) - 1))

Base.isinf(x::T) where T<:IEEEFloat = reinterpret(Unsigned, x) & exponent_mask(T) == exponent_mask(T) && iszero(reinterpret(Unsigned, x) & significand_mask(T))
Base.isnan(x::T) where T<:IEEEFloat = reinterpret(Unsigned, x) & exponent_mask(T) == exponent_mask(T) && !iszero(reinterpret(Unsigned, x) & significand_mask(T))

Base.floatmax(::Type{T}) where T<:IEEEFloat = reinterpret(T, exponent_mask(T) - 0x01 << significand_bits(T) | significand_mask(T))

const Float8_E3M4 = IEEEFloat{1,3,4}
const Float8_E4M3 = IEEEFloat{1,4,3}
const Float8_E5M2 = IEEEFloat{1,5,2}
const Float6_E2M3 = IEEEFloat{1,2,3}
const Float6_E3M2 = IEEEFloat{1,3,2}
const Float4_E2M1 = IEEEFloat{1,2,1}
