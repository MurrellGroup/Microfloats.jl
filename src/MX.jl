abstract type MXFloatVariant <: FloatVariant end

const MXFloat{S,E,M} = Microfloat{S,E,M,MXFloatVariant}

const MX_E5M2 = MXFloat{1,5,2}
const MX_E4M3 = MXFloat{1,4,3}
const MX_E3M2 = MXFloat{1,3,2}
const MX_E2M3 = MXFloat{1,2,3}
const MX_E2M1 = MXFloat{1,2,1}
const MX_E8M0 = MXFloat{0,8,0}

export MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1, MX_E8M0

hasinf(::Type{MX_E4M3}) = false
nan(::Type{T}) where T<:MX_E4M3 = reinterpret(T, exponent_mask(T) | significand_mask(T))
Base.isnan(x::T) where T<:MX_E4M3 = reinterpret(Unsigned, x) & ~sign_mask(T) == (exponent_mask(T) | significand_mask(T))
Base.floatmax(::Type{T}) where T<:MX_E4M3 = reinterpret(T, exponent_mask(T) | (significand_mask(T) - 0x01))

hasinf(::Type{MX_E3M2}) = false
hasnan(::Type{MX_E3M2}) = false
Base.floatmax(::Type{T}) where T<:MX_E3M2 = reinterpret(T, exponent_mask(T) | significand_mask(T))

hasinf(::Type{MX_E2M3}) = false
hasnan(::Type{MX_E2M3}) = false
Base.floatmax(::Type{T}) where T<:MX_E2M3 = reinterpret(T, exponent_mask(T) | significand_mask(T))

hasinf(::Type{MX_E2M1}) = false
hasnan(::Type{MX_E2M1}) = false
Base.floatmax(::Type{T}) where T<:MX_E2M1 = reinterpret(T, exponent_mask(T) | significand_mask(T))

hasinf(::Type{MX_E8M0}) = false
nan(::Type{T}) where T<:MX_E8M0 = reinterpret(T, 0xff)
Base.isnan(x::T) where T<:MX_E8M0 = reinterpret(Unsigned, x) == 0xff
Base.floatmax(::Type{T}) where T<:MX_E8M0 = reinterpret(T, 0xfe)
