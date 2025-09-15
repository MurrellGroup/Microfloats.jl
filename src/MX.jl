abstract type MX <: Finite end

const MXFloat{S,E,M} = Microfloat{S,E,M,MX}

const MX_E5M2 = MXFloat{1,5,2}
const MX_E4M3 = MXFloat{1,4,3}
const MX_E3M2 = MXFloat{1,3,2}
const MX_E2M3 = MXFloat{1,2,3}
const MX_E2M1 = MXFloat{1,2,1}
const MX_E8M0 = MXFloat{0,8,0}

hasinf(::Type{MX_E5M2}) = true
hasnan(::Type{MX_E5M2}) = true
inf(::Type{T}) where T<:MX_E5M2 = reinterpret(T, exponent_mask(T))
nan(::Type{T}) where T<:MX_E5M2 = reinterpret(T, exponent_mask(T) | 0x01 << (significand_bits(T) - 1))
Base.isinf(x::T) where T<:MX_E5M2 = hasinf(T) && reinterpret(Unsigned, x) & exponent_mask(T) == exponent_mask(T) && iszero(reinterpret(Unsigned, x) & significand_mask(T))
Base.isnan(x::T) where T<:MX_E5M2 = reinterpret(Unsigned, x) & exponent_mask(T) == exponent_mask(T) && !iszero(reinterpret(Unsigned, x) & significand_mask(T))
Base.floatmax(::Type{T}) where T<:MX_E5M2 = reinterpret(T, exponent_mask(T) - 0x01 << significand_bits(T) | significand_mask(T))

hasnan(::Type{MX_E4M3}) = true
nan(::Type{T}) where T<:MX_E4M3 = reinterpret(T, exponent_mask(T) | significand_mask(T))
Base.isnan(x::T) where T<:MX_E4M3 = reinterpret(Unsigned, x) & ~sign_mask(T) == (exponent_mask(T) | significand_mask(T))
Base.floatmax(::Type{T}) where T<:MX_E4M3 = reinterpret(T, exponent_mask(T) | (significand_mask(T) - 0x01))

hasnan(::Type{MX_E8M0}) = true
nan(::Type{T}) where T<:MX_E8M0 = reinterpret(T, 0xff)
Base.isnan(x::T) where T<:MX_E8M0 = reinterpret(Unsigned, x) == 0xff
Base.floatmax(::Type{T}) where T<:MX_E8M0 = reinterpret(T, 0xfe)
