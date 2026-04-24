import Random

# Draw a Uniform(0,1) value for Microfloats via the CloseOpen01 sampler
function Base.rand(rng::Random.AbstractRNG, ::Random.SamplerTrivial{Random.CloseOpen01{T}, T}) where T<:Microfloat
    return T(rand(rng, Random.CloseOpen01(Float32)))
end

# Standard normal sampling for signed Microfloats
Base.randn(::Random.AbstractRNG, ::Type{T}) where T<:Microfloat =
    throw(ArgumentError("randn is undefined for unsigned microfloats (must have 1 sign bit)"))
function Base.randn(rng::Random.AbstractRNG, ::Type{T}) where T<:Microfloat{1}
    z = randn(rng, Float32)
    b = Float32(floatmax(T))
    return T(clamp(z, -b, b))
end
