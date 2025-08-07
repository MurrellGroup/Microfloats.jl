primitive type Microfloat{S,E,M,V} <: AbstractFloat 8 end

const SignedMicrofloat = Microfloat{1}
const UnsignedMicrofloat = Microfloat{0}

function Microfloat(S::Int, E::Int, M::Int; variant::Symbol=:IEEE)
    S in (0, 1) || throw(ArgumentError("sign bit must be 0 or 1"))
    E >= 0 || throw(ArgumentError("number of exponent bits must be non-negative"))
    M >= 0 || throw(ArgumentError("number of mantissa bits must be non-negative"))
    0 < S + E + M <= 8 || throw(ArgumentError("total number of bits must be between 1 and 8"))
    Microfloat{S,E,M,variant}
end

Microfloat{S}(E::Int, M::Int; kws...) where S = Microfloat(S, E, M; kws...)

uint(::Type{T}) where T<:Microfloat = UInt8
n_sign_bits(::Type{T}) where {S,T<:Microfloat{S}} = S
n_exponent_bits(::Type{T}) where {E,T<:Microfloat{<:Any,E}} = E
n_mantissa_bits(::Type{T}) where {M,T<:Microfloat{<:Any,<:Any,M}} = M

Base.hash(x::Microfloat, h::UInt) = hash(Float32(x), h)

inf(::Type{T}) where T<:Microfloat = reinterpret(T, exponent_mask(T))
nan(::Type{T}) where T<:Microfloat = reinterpret(T, bit_ones(n_exponent_bits(T) + 1) << (exponent_offset(T) - has_mantissa(T)))

Base.isinf(x::T) where T<:Microfloat = x === inf(T)
Base.isnan(x::T) where T<:Microfloat = only_exponent(x) === exponent_mask(T) && !iszero(only_mantissa(x))

Base.zero(::Type{T}) where T<:Microfloat = reinterpret(T, 0x00)
Base.one(::Type{T}) where T<:Microfloat = reinterpret(T, bit_ones(n_exponent_bits(T) - 1) << exponent_offset(T))

Base.eps(x::Microfloat) = max(x-prevfloat(x), nextfloat(x)-x)
Base.eps(T::Type{<:Microfloat}) = eps(one(T))

Base.floatmin(::Type{T}) where T<:Microfloat = reinterpret(T, bit_ones(1) << exponent_offset(T))
Base.floatmax(::Type{T}) where T<:Microfloat = reinterpret(T, bit_ones(n_exponent_bits(T) - 1) << (exponent_offset(T) + 1) | mantissa_mask(T))

Base.typemin(::Type{T}) where T<:Microfloat = -inf(T)
Base.typemin(::Type{T}) where T<:UnsignedMicrofloat = zero(T)

Base.typemax(::Type{T}) where T<:Microfloat = inf(T)

Base.abs(x::T) where T<:Microfloat = reinterpret(T, reinterpret(UInt8, x) & ~sign_mask(T))
Base.iszero(x::T) where T<:Microfloat = abs(x) === zero(T)
Base.:(-)(x::T) where T<:Microfloat = reinterpret(T, sign_mask(T) ⊻ reinterpret(UInt8, x))
Base.Bool(x::T) where T<:Microfloat = iszero(x) ? false : isone(x) ? true : throw(InexactError(:Bool, Bool, x))

# https://github.com/JuliaLang/julia/blob/46c2a5c7e1f970e83b408b6ddcba49aaa31d8329/base/float.jl#L799-L801
Base.precision(::Type{T}) where T<:Microfloat = n_mantissa_bits(T) + 1

Base.signbit(x::Microfloat) = x !== abs(x)
Base.sign(x::Microfloat) = ifelse(isnan(x) || iszero(x), x, ifelse(signbit(x), -one(x), one(x)))

Base.round(x::Microfloat, r::RoundingMode) = reinterpret(typeof(x), round(Float32(x), r))

function Base.nextfloat(x::T) where T<:Microfloat
    if isnan(x) || x === inf(T)
        return x
    elseif iszero(x)
        return reinterpret(T, bit_ones(1) << mantissa_offset(T))
    elseif ispositive(x)
        return reinterpret(T, reinterpret(UInt8, x) + bit_ones(1) << mantissa_offset(T))
    else
        return reinterpret(T, reinterpret(UInt8, x) - bit_ones(1) << mantissa_offset(T))
    end
end

function Base.prevfloat(x::T) where T<:Microfloat
    if isnan(x) || x === -inf(T)
        return x
    elseif iszero(x)
        return reinterpret(T, sign_mask(T) | (bit_ones(1) << mantissa_offset(T)))
    elseif ispositive(x)
        return reinterpret(T, reinterpret(UInt8, x) - bit_ones(1) << mantissa_offset(T))
    else
        return reinterpret(T, reinterpret(UInt8, x) + bit_ones(1) << mantissa_offset(T))
    end
end

Base.promote_rule(::Type{<:Microfloat},::Type{Float16}) = Float16
Base.promote_rule(::Type{<:Microfloat},::Type{Float32}) = Float32
Base.promote_rule(::Type{<:Microfloat},::Type{Float64}) = Float64
Base.promote_rule(T::Type{<:Microfloat},::Type{<:Integer}) = T

# make work with other types
# reduce branching
function Base.:(==)(x::T, y::T) where T<:Microfloat
    (isnan(x) || isnan(y)) && return false
    (iszero(x) && iszero(y)) && return true
    return x === y
end
