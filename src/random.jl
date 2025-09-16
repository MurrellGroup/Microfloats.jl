import Random

# Draw a Uniform(0,1) value for Microfloats via the CloseOpen01 sampler
function Base.rand(rng::Random.AbstractRNG, ::Random.SamplerTrivial{Random.CloseOpen01{T}, T}) where T<:Microfloat
    return T(rand(rng, Random.CloseOpen01(Float32)))
end

# Standard normal sampling for signed Microfloats
function Base.randn(rng::Random.AbstractRNG, ::Type{T}) where {S,E,M,V,T<:Microfloat{S,E,M,V}}
    S == 0 && throw(ArgumentError("randn is undefined for unsigned microfloats (no sign bit)"))
    z = randn(rng, Float32)
    b = Float32(floatmax(T))
    return T(clamp(z, -b, b))
end
