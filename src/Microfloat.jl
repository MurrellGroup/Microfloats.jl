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

hasinf(::Type{T}) where T<:Microfloat = true
hasnan(::Type{T}) where T<:Microfloat = true

inf(::Type{T}) where T<:Microfloat = hasinf(T) ? reinterpret(T, exponent_mask(T)) : throw(DomainError(T, "$T has no Inf"))
nan(::Type{T}) where T<:Microfloat = hasnan(T) ? reinterpret(T, exponent_mask(T) | 0x01 << (significand_bits(T) - 1)) : throw(DomainError(T, "$T has no NaN"))

Base.isinf(x::T) where T<:Microfloat = hasinf(T) && reinterpret(Unsigned, x) & exponent_mask(T) == exponent_mask(T) && iszero(reinterpret(Unsigned, x) & significand_mask(T))
Base.isnan(x::T) where T<:Microfloat = hasnan(T) && reinterpret(Unsigned, x) & exponent_mask(T) == exponent_mask(T) && !iszero(reinterpret(Unsigned, x) & significand_mask(T))

Base.typemin(::Type{T}) where T<:Microfloat = hasinf(T) ? -inf(T) : -floatmax(T)
Base.typemin(::Type{T}) where T<:Microfloat{0} = zero(T)
Base.typemax(::Type{T}) where T<:Microfloat = hasinf(T) ? inf(T) : floatmax(T)

Base.zero(::Type{T}) where T<:Microfloat = reinterpret(T, 0x00)
Base.one(::Type{T}) where T<:Microfloat = T(true)

Base.eps(x::Microfloat) = max(x-prevfloat(x), nextfloat(x)-x)
Base.eps(T::Type{<:Microfloat}) = eps(one(T))

Base.floatmin(::Type{T}) where T<:Microfloat = exponent_bits(T) > 1 ? reinterpret(T, one(UInt8) << significand_bits(T)) : throw(DomainError(T, "$T has no normal numbers"))
Base.floatmax(::Type{T}) where T<:Microfloat = reinterpret(T, exponent_mask(T) - 0x01 << significand_bits(T) | significand_mask(T))

Base.abs(x::T) where T<:Microfloat = reinterpret(T, reinterpret(Unsigned, x) & ~sign_mask(T))
Base.iszero(x::T) where T<:Microfloat = abs(x) === zero(T)
Base.:(-)(x::T) where T<:Microfloat = reinterpret(T, sign_mask(T) ⊻ reinterpret(Unsigned, x))
Base.Bool(x::T) where T<:Microfloat = iszero(x) ? false : isone(x) ? true : throw(InexactError(:Bool, Bool, x))

Base.precision(::Type{T}) where T<:Microfloat = significand_bits(T) + 1

Base.sign(x::Microfloat) = ifelse(isnan(x) | iszero(x), x, ifelse(signbit(x), -one(x), one(x)))

Base.round(x::T, r::RoundingMode; kws...) where T<:Microfloat = T(round(Float32(x), r; kws...))

Base.issubnormal(x::T) where T<:Microfloat = 0x00 < (reinterpret(Unsigned, x) & ~sign_mask(T)) <= (0x01 << significand_bits(T)) - 0x01

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
        return reinterpret(T, sign_mask(T) | 0x01)
    elseif ispositive(x)
        return reinterpret(T, reinterpret(Unsigned, x) - 0x01)
    else
        return reinterpret(T, reinterpret(Unsigned, x) + 0x01)
    end
end

Base.decompose(x::T) where T<:Microfloat = Base.decompose(BFloat16(x))

Base.promote_rule(::Type{M}, ::Type{T}) where {M<:Microfloat,T<:Union{BFloat16,Float16,Float32,Float64}} = T
Base.promote_rule(::Type{M}, ::Type{T}) where {M<:Microfloat,T<:Integer} = M