import Random

# Minimal hooks into Random's sampling API

# Teach Random how to materialize the trivial sampler for Uniform(0,1) on Microfloats
Random.Sampler(::Type{RNG}, sp::Random.SamplerTrivial{Random.CloseOpen01{T}, T}, ::Val{1}) where {RNG<:Random.AbstractRNG, T<:Microfloat} = sp
Random.Sampler(rng::Random.AbstractRNG, sp::Random.SamplerTrivial{Random.CloseOpen01{T}, T}, ::Val{1}) where {T<:Microfloat} = sp

# Draw a Uniform(0,1) value for Microfloats via the CloseOpen01 sampler
function Random.rand(rng::Random.AbstractRNG, ::Random.SamplerTrivial{Random.CloseOpen01{T}, T}) where {T<:Microfloat}
    return T(Random.rand(rng, Random.CloseOpen01(Float32)))
end

# Standard normal sampling for signed Microfloats
function Random.randn(rng::Random.AbstractRNG, ::Type{T}) where {S,E,M,V,T<:Microfloat{S,E,M,V}}
    S == 0 && throw(ArgumentError("randn is undefined for unsigned microfloats (no sign bit)"))
    z = Random.randn(rng, Float32)
    b = Float32(floatmax(T))
    return T(clamp(z, -b, b))
end
