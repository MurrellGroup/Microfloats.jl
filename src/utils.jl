float_bits(::Type{Float64}) = (1, 11, 52)
float_bits(::Type{Float32}) = (1, 8, 23)
float_bits(::Type{Float16}) = (1, 5, 10)
float_bits(::Type{BFloat16}) = (1, 8, 7)

"""
    bitwidth(::Type{<:AbstractFloat})

Return the total number of utilized bits — the sum of sign, exponent, and
significand bits. For Base float types this matches the storage size; for
[`Microfloat`](@ref) subtypes narrower than 8 bits, the value is smaller
than the underlying byte.

# Examples
```jldoctest
julia> Microfloats.bitwidth(Float4_E2M1FN)
4

julia> Microfloats.bitwidth(Float64)
64
```
"""
bitwidth(::Type{T}) where T<:AbstractFloat = sum(float_bits(T))

"""
    sign_bits(::Type{<:AbstractFloat}) -> Int

Return the number of sign bits (`0` for unsigned microfloats, `1` otherwise).

# Examples
```jldoctest
julia> Microfloats.sign_bits(Float8_E4M3)
1

julia> Microfloats.sign_bits(Float8_E8M0FNU)
0
```
"""
sign_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[1]

"""
    exponent_bits(::Type{<:AbstractFloat}) -> Int

Return the number of exponent bits.

# Examples
```jldoctest
julia> Microfloats.exponent_bits(Float8_E4M3)
4

julia> Microfloats.exponent_bits(Float32)
8
```
"""
exponent_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[2]

"""
    significand_bits(::Type{<:AbstractFloat}) -> Int

Return the number of significand / mantissa bits (excluding the implicit
leading 1 of normal IEEE values).

# Examples
```jldoctest
julia> Microfloats.significand_bits(Float8_E4M3)
3

julia> Microfloats.significand_bits(Float64)
52
```
"""
significand_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[3]

exponent_bias(::Type{T}) where T<:AbstractFloat = 2^(exponent_bits(T) - 1) - 1
