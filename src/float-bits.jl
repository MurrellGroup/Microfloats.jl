n_sign_bits(::Type{<:AbstractFloat}) = 1
n_exponent_bits(::Type{Float32}) = 8
n_mantissa_bits(::Type{Float32}) = 23

uint(::Type{Float32}) = UInt32
uint(::Type{T}) where T<:Unsigned = T
as_uint(x::T) where T<:AbstractFloat = reinterpret(uint(T), x)
bit_ones(N, T=UInt8) = (one(uint(T)) << N) - one(uint(T))

n_bits(::Type{T}) where T<:AbstractFloat = n_sign_bits(T) + n_exponent_bits(T) + n_mantissa_bits(T)

mantissa_offset(::Type{T}) where T<:AbstractFloat = 0
exponent_offset(::Type{T}) where T<:AbstractFloat = n_mantissa_bits(T) + mantissa_offset(T)
sign_offset(::Type{T}) where T<:AbstractFloat = n_exponent_bits(T) + exponent_offset(T)

sign_mask(::Type{T}) where T<:AbstractFloat = bit_ones(n_sign_bits(T), T) << sign_offset(T)
exponent_mask(::Type{T}) where T<:AbstractFloat = bit_ones(n_exponent_bits(T), T) << exponent_offset(T)
mantissa_mask(::Type{T}) where T<:AbstractFloat = bit_ones(n_mantissa_bits(T), T) << mantissa_offset(T)

only_sign(x::T) where T<:AbstractFloat = as_uint(x) & sign_mask(T)
only_exponent(x::T) where T<:AbstractFloat = as_uint(x) & exponent_mask(T)
only_mantissa(x::T) where T<:AbstractFloat = as_uint(x) & mantissa_mask(T)

right_aligned_sign(x::T) where T<:AbstractFloat = only_sign(x) >> sign_offset(T)
right_aligned_exponent(x::T) where T<:AbstractFloat = only_exponent(x) >> exponent_offset(T)
right_aligned_mantissa(x::T) where T<:AbstractFloat = only_mantissa(x) >> mantissa_offset(T)

exponent_bias(::Type{T}) where T<:AbstractFloat = UInt32(2^(n_exponent_bits(T) - 1) - 1)

# right_aligned_sign_mask(::Type{T}) where T<:AbstractFloat = bit_ones(n_sign_bits(T), T)
right_aligned_exponent_mask(::Type{T}) where T<:AbstractFloat = bit_ones(n_exponent_bits(T), T)
right_aligned_mantissa_mask(::Type{T}) where T<:AbstractFloat = bit_ones(n_mantissa_bits(T), T)

# has_sign(::Type{T}) where T<:AbstractFloat = n_sign_bits(T) > 0
# has_exponent(::Type{T}) where T<:AbstractFloat = n_exponent_bits(T) > 0
has_mantissa(::Type{T}) where T<:AbstractFloat = n_mantissa_bits(T) > 0

ispositive(x::T) where T<:AbstractFloat = as_uint(x) & sign_mask(T) === zero(uint(T))
# isnegative(x::T) where T<:AbstractFloat = as_uint(x) & sign_mask(T) !== zero(uint(T))
