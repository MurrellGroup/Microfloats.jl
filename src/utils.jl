float_bits(::Type{Float64}) = (1, 11, 52)
float_bits(::Type{Float32}) = (1, 8, 23)
float_bits(::Type{Float16}) = (1, 5, 10)
float_bits(::Type{BFloat16}) = (1, 8, 7)

"""
    bitwidth(::Type{<:AbstractFloat})

Returns the number of utilized bits.

```jldoctest
julia> Microfloats.bitwidth(Float4_E2M1FN)
4
```
"""
bitwidth(::Type{T}) where T<:AbstractFloat = sum(float_bits(T))

"""
    sign_bits(::Type{<:AbstractFloat})

Return the number of sign bits (between 0 or 1).
"""
sign_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[1]

"""
    exponent_bits(::Type{<:AbstractFloat})

Return the number of exponent bits.
"""
exponent_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[2]

"""
    significand_bits(::Type{<:AbstractFloat})

Return the number of significand / mantissa bits.
"""
significand_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[3]

exponent_bias(::Type{T}) where T<:AbstractFloat = 2^(exponent_bits(T) - 1) - 1
