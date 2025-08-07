bias(::Type{T}) where T<:Microfloat = UInt32(2^(n_exponent_bits(T) - 1) - 1)
bias_difference(::Type{T}) where T<:Microfloat = UInt32(127 - bias(T))

include("to_microfloat.jl")
include("from_microfloat.jl")

(::Type{T})(x::Microfloat) where T<:Number = T(Float32(x))
(::Type{T})(x::Number) where T<:Microfloat = T(Float32(x))
