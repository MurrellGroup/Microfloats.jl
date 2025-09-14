import Base:
    reinterpret,
    exponent_bits, significand_bits,
    sign_mask, exponent_mask, significand_mask,
    signbit, exponent, exponent_bias

sign_bits(::Type{<:AbstractFloat}) = 1
total_bits(::Type{T}) where T<:AbstractFloat = sign_bits(T) + exponent_bits(T) + significand_bits(T)

abstract type FloatVariant end

primitive type Microfloat{S,E,M,V<:FloatVariant} <: AbstractFloat 8 end

reinterpret(::Type{Unsigned}, x::Microfloat) = reinterpret(UInt8, x)

sign_bits(::Type{<:Microfloat{S}}) where S = S
exponent_bits(::Type{<:Microfloat{<:Any,E}}) where E = E
significand_bits(::Type{<:Microfloat{<:Any,<:Any,M}}) where M = M
variant(::Type{<:Microfloat{<:Any,<:Any,<:Any,V}}) where V = V

sign_mask(::Type{T}) where T<:Microfloat = (0x01 << sign_bits(T) - 0x01) << (exponent_bits(T) + significand_bits(T))
exponent_mask(::Type{T}) where T<:Microfloat = (0x01 << exponent_bits(T) - 0x01) << significand_bits(T)
significand_mask(::Type{T}) where T<:Microfloat = 0x01 << significand_bits(T) - 0x01

signbit(x::Microfloat) = sign_bits(typeof(x)) > 0 && reinterpret(Unsigned, x) & sign_mask(typeof(x)) == sign_mask(typeof(x))
exponent(x::Microfloat) = Int((reinterpret(Unsigned, x) & exponent_mask(typeof(x))) >> significand_bits(typeof(x)))
exponent_bias(::Type{T}) where T<:Microfloat = 2^(exponent_bits(T) - 1) - 1

inf(::Type{T}) where T<:Microfloat = reinterpret(T, exponent_mask(T))
nan(::Type{T}) where T<:Microfloat = has_mantissa(T) ?
    reinterpret(T, exponent_mask(T) | (0x01 << mantissa_offset(T))) :
    throw(DomainError(T, "$T has no NaN values"))

# TODO: Base.issubnormal

Base.isinf(x::T) where T<:Microfloat = iszero(reinterpret(Unsigned, x) ⊻ exponent_mask(T)) & iszero(reinterpret(Unsigned, x) & significand_mask(T))
Base.isnan(x::T) where T<:Microfloat = iszero(reinterpret(Unsigned, x) ⊻ exponent_mask(T)) & !iszero(reinterpret(Unsigned, x) & significand_mask(T))

Base.zero(::Type{T}) where T<:Microfloat = reinterpret(T, 0x00)
Base.one(::Type{T}) where T<:Microfloat = T(1.0f0)

Base.eps(x::Microfloat) = max(x-prevfloat(x), nextfloat(x)-x)
Base.eps(T::Type{<:Microfloat}) = eps(one(T))

Base.floatmin(::Type{T}) where T<:Microfloat = exponent_bits(T) > 1 ? reinterpret(T, one(UInt8) << significand_bits(T)) : throw(DomainError(T, "$T has no normal numbers"))
Base.floatmax(::Type{T}) where T<:Microfloat = reinterpret(T, (0x01 << exponent_bits(T) - 0x02) << significand_bits(T) | significand_mask(T))

Base.typemin(::Type{T}) where T<:Microfloat = -inf(T)
Base.typemin(::Type{T}) where T<:Microfloat{0} = zero(T)

Base.typemax(::Type{T}) where T<:Microfloat = inf(T)

Base.abs(x::T) where T<:Microfloat = reinterpret(T, reinterpret(UInt8, x) & ~sign_mask(T))
Base.iszero(x::T) where T<:Microfloat = abs(x) === zero(T)
Base.:(-)(x::T) where T<:Microfloat = reinterpret(T, sign_mask(T) ⊻ reinterpret(UInt8, x))
Base.Bool(x::T) where T<:Microfloat = iszero(x) ? false : isone(x) ? true : throw(InexactError(:Bool, Bool, x))

Base.precision(::Type{T}) where T<:Microfloat = significand_bits(T) + 1

Base.sign(x::Microfloat) = ifelse(isnan(x) || iszero(x), x, ifelse(signbit(x), -one(x), one(x)))

Base.round(x::T, r::RoundingMode; kws...) where T<:Microfloat = T(round(Float32(x), r; kws...))
