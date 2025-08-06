primitive type Microfloat{N,E,M,S,OnlyFinite} <: AbstractFloat 8 end

@inline default_onlyfinite() = false
@inline default_mantissa(N, E) = N - E - (N != E)
@inline default_sign(N, E, M) = N - E - M

@inline default(T::Type{Microfloat{N,E,M,S,OnlyFinite}}) where {N,E,M,S,OnlyFinite} = T
@inline default(T::Type{Microfloat{N,E,M,S}}) where {N,E,M,S} = T{default_onlyfinite()}
@inline default(T::Type{Microfloat{N,E,M}}) where {N,E,M} = default(T{default_sign(N,E,M)})
@inline default(T::Type{Microfloat{N,E}}) where {N,E} = default(T{default_mantissa(N,E)})

const SignlessMicrofloat{N,E,M} = Microfloat{N,E,M,0}
@inline default(T::Type{SignlessMicrofloat{N,E}}) where {N,E} = default(T{N-E})

const FiniteMicrofloat{N,E,M,S} = Microfloat{N,E,M,S,true}
@inline default(T::Type{FiniteMicrofloat{N,E,M}}) where {N,E,M} = default(T{default_sign(N,E,M)})
@inline default(T::Type{FiniteMicrofloat{N,E}}) where {N,E} = default(T{default_mantissa(N,E)})

(T::Type{<:Number})(x::Microfloat) = T(Float32(x))
(T::Type{<:Microfloat})(x::Number) = T(Float32(x))

@inline function (T::Type{<:Microfloat})(x)
    new_T = default(T)
    T == new_T && throw(MethodError(T, (x,)))
    new_T(x)
end

@inline _reinterpret(T, x) = reinterpret(T, x)
@inline _reinterpret(T::Type{<:Microfloat}, x) = reinterpret(default(T), x)

@inline Microfloat{N,E,M,S}(x::Number) where {N,E,M,S} = Microfloat{N,E,M,S,false}(x)
@inline Microfloat{N,E,M}(x::Number) where {N,E,M} = Microfloat{N,E,M,N-E-M}(x)
@inline Microfloat{N,E}(x::Number) where {N,E} = Microfloat{N,E,default_mantissa(N,E)}(x)

@inline n_bits(::Type{<:Microfloat{N}}) where N = N
@inline n_padding_bits(::Type{<:Microfloat{N}}) where N = 8 - N
@inline n_exponent_bits(::Type{<:Microfloat{N,E}}) where {N,E} = E
@inline n_mantissa_bits(::Type{<:Microfloat{N,E}}) where {N,E} = default_mantissa(N, E)
@inline n_mantissa_bits(::Type{<:Microfloat{N,E,M}}) where {N,E,M} = M
@inline n_sign_bits(::Type{<:Microfloat{N,E,M}}) where {N,E,M} = default_sign(N, E, M)
@inline n_sign_bits(::Type{<:Microfloat{N,E,M,S}}) where {N,E,M,S} = S

@inline mantissa_offset(T::Type{<:Microfloat}) = n_padding_bits(T)
@inline exponent_offset(T::Type{<:Microfloat}) = n_mantissa_bits(T) + n_padding_bits(T)

@inline bit_ones(N, T=UInt8) = (one(T) << N) - one(T)

sign_mask(T::Type{<:Microfloat}) = bit_ones(n_sign_bits(T)) << 7
not_sign_mask(T::Type{<:Microfloat}) = bit_ones(n_bits(T) - n_sign_bits(T)) << mantissa_offset(T)
exponent_mask(T::Type{<:Microfloat}) = bit_ones(n_exponent_bits(T)) << exponent_offset(T)
mantissa_mask(T::Type{<:Microfloat}) = bit_ones(n_mantissa_bits(T)) << mantissa_offset(T)

sign_mask(::Type{Float32}) = 0x8000_0000
exponent_mask(::Type{Float32}) = 0x7f80_0000
mantissa_mask(::Type{Float32}) = 0x007f_ffff

eps(x::Microfloat) = max(x-prevfloat(x), nextfloat(x)-x)
eps(T::Type{<:Microfloat}) = eps(one(T))

_onlyfinite(::Type{<:Microfloat{N,E,M,S,OnlyFinite}}) where {N,E,M,S,OnlyFinite} = OnlyFinite

_has_mantissa(T::Type{<:Microfloat}) = n_mantissa_bits(T) > 0

@inline function _bits_inf(::Type{T}) where T<:Microfloat
    if _onlyfinite(T)               # finite-only formats (e.g. MX-FP4)
        return bit_ones(n_exponent_bits(T)-1) << (exponent_offset(T)+1) |
               mantissa_mask(T)   # → saturate to typemax
    else
        return bit_ones(n_exponent_bits(T)) << exponent_offset(T)
    end
end

@inline function _bits_nan(::Type{T}) where T<:Microfloat
    if !_has_mantissa(T) || _onlyfinite(T)
        return _bits_inf(T)       # NaN impossible → fall back to Inf
    else
        return _bits_inf(T) |      # all-ones exponent
               (one(UInt8) << mantissa_offset(T))  # set quiet-NaN bit
    end
end

@inline inf(T::Type{<:Microfloat}) = _reinterpret(T, _bits_inf(T))
@inline nan(T::Type{<:Microfloat}) = _reinterpret(T, _bits_nan(T))

@inline isinf(x::Microfloat) = (reinterpret(UInt8,x) & exponent_mask(typeof(x))) ==
                               exponent_mask(typeof(x)) &&
                               (reinterpret(UInt8,x) & mantissa_mask(typeof(x))) == 0

@inline isnan(x::Microfloat) = (reinterpret(UInt8,x) & exponent_mask(typeof(x))) ==
                               exponent_mask(typeof(x)) &&
                               (reinterpret(UInt8,x) & mantissa_mask(typeof(x))) != 0

floatmin(T::Type{<:Microfloat}) = _reinterpret(T, bit_ones(1) << exponent_offset(T))
floatmax(T::Type{<:Microfloat}) = _reinterpret(T, bit_ones(n_exponent_bits(T) - 1) << (exponent_offset(T) + 1) | mantissa_mask(T))

typemin(T::Type{<:Microfloat}) = -inf(T)
typemax(T::Type{<:Microfloat}) = inf(T)

typemin(T::Type{<:SignlessMicrofloat}) = floatmin(T)
typemin(T::Type{<:FiniteMicrofloat}) = floatmin(T)
typemax(T::Type{<:FiniteMicrofloat}) = floatmax(T)

one(T::Type{<:Microfloat}) = _reinterpret(T, bit_ones(n_exponent_bits(T) - 1) << exponent_offset(T))
zero(T::Type{<:Microfloat}) = _reinterpret(T, 0x00)

one(x::Microfloat) = one(typeof(x))
zero(x::Microfloat) = zero(typeof(x))

iszero(x::Microfloat) = x == zero(typeof(x))
-(x::Microfloat) = reinterpret(typeof(x), sign_mask(typeof(x)) ⊻ reinterpret(UInt8, x))
Bool(x::Microfloat) = iszero(x) ? false : isone(x) ? true : throw(InexactError(:Bool, Bool, x))

abs(x::Microfloat) = reinterpret(typeof(x), reinterpret(UInt8, x) & ~sign_mask(typeof(x)))

precision(x::Type{<:Microfloat}) = n_mantissa_bits(x)

signbit(x::Microfloat) = reinterpret(UInt8, x) > ~sign_mask(typeof(x))
sign(x::Microfloat) = ifelse(isnan(x) || iszero(x), x, ifelse(signbit(x), -one(x), one(x)))

first_mantissa_bit_mask(T::Type{<:Microfloat}) = one(UInt32) << (exponent_offset(T) - 1)

mantissa_bit_shift(T::Type{<:Microfloat}) = 23 - n_mantissa_bits(T)

bias(T::Type{<:Microfloat}) = UInt32(2^(n_exponent_bits(T) - 1) - 1)
bias_difference(T::Type{<:Microfloat}) = UInt32(127 - bias(T))

exp_bits_all_one(T::Type{<:Microfloat}) = bit_ones(n_exponent_bits(T), UInt32)

round(x::Microfloat, r::RoundingMode) = reinterpret(typeof(x), round(Float32(x), r))

function ==(x::T, y::T) where T<:Microfloat
    if isnan(x) || isnan(y)     # Alternatively, For Float16: (ix|iy)&0x7fff > 0x7c00
        return false
    end
    return reinterpret(UInt8, x) == reinterpret(UInt8, y)
end

for op in (:<, :<=, :isless)
    @eval ($op)(a::Microfloat, b::Real) = ($op)(Float32(a), Float32(b))
    @eval ($op)(a::Real, b::Microfloat) = ($op)(Float32(a), Float32(b))
    @eval ($op)(a::Microfloat, b::Microfloat) = ($op)(Float32(a), Float32(b))
end

for op in (:+, :-, :*, :/, :\, :^)
    @eval ($op)(a::Microfloat, b::Number) = promote_type(typeof(a), typeof(b))(($op)(Float32(a), Float32(b)))
    @eval ($op)(a::Number, b::Microfloat) = promote_type(typeof(a), typeof(b))(($op)(Float32(a), Float32(b)))
    @eval ($op)(a::Microfloat, b::Microfloat) = promote_type(typeof(a), typeof(b))(($op)(Float32(a), Float32(b)))
end

for func in (:sin,:cos,:tan,:asin,:acos,:atan,:sinh,:cosh,:tanh,:asinh,:acosh,
    :atanh,:exp,:exp2,:exp10,:expm1,:log,:log2,:log10,:sqrt,:cbrt,:log1p)
    @eval $func(a::T) where T<:Microfloat = T($func(Float32(a)))
end

for func in (:atan,:hypot)
    @eval $func(a::T, b::T) where T<:Microfloat = T($func(Float32(a),Float32(b)))
end

for func in (:frexp,:ldexp)
    @eval $func(a::T, b::Int) where T<:Microfloat = T($func(Float32(a),b))
end

for func in (:modf,:mod2pi)
    @eval $func(a::T) where T<:Microfloat = T($func(Float32(a)))
end

function Base.show(io::IO, x::T) where T <: Microfloat
    if isnan(x)
        print(io, T, "(NaN32)")
    elseif isinf(x)
        # use the format-specific sign mask
        if n_sign_bits(T) > 0 && (reinterpret(UInt8, x) & sign_mask(T)) != 0
            print(io, T, "(-Inf32)")
        else
            print(io, T, "(Inf32)")
        end
    else
        io2 = IOBuffer()
        print(io2, repr(Float32(x)))
        f = String(take!(io2))
        print(io, T, "("*f*")")
    end
end

function nextfloat(x::T) where T<:Microfloat
    if isnan(x) || x == inf(T)
        return x
    elseif x == -zero(T)
        return reinterpret(T, 0x01)
    elseif UInt8(x) < 0x80  # positive numbers
        return reinterpret(T, reinterpret(UInt8, x) + 0x01)
    else                    # negative numbers
        return reinterpret(T, reinterpret(UInt8, x) - 0x01)
    end
end

function prevfloat(x::T) where T<:Microfloat
    if isnan(x) || x == -inf(T)
        return x
    elseif x == zero(T)
        return reinterpret(T, 0x81)
    elseif reinterpret(UInt8, x) < 0x80
        return reinterpret(T, reinterpret(UInt8, x) - 0x01)
    else
        return reinterpret(T, reinterpret(UInt8, x) + 0x01)
    end
end

Base.promote_rule(::Type{<:Microfloat},::Type{Float16}) = Float16
Base.promote_rule(::Type{<:Microfloat},::Type{Float32}) = Float32
Base.promote_rule(::Type{<:Microfloat},::Type{Float64}) = Float64
Base.promote_rule(T::Type{<:Microfloat},::Type{<:Integer}) = T
