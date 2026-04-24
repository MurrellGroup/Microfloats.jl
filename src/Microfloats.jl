module Microfloats

using Republic

@reexport import BFloat16s: BFloat16

float_bits(::Type{Float64}) = (1, 11, 52)
float_bits(::Type{Float32}) = (1, 8, 23)
float_bits(::Type{Float16}) = (1, 5, 10)
float_bits(::Type{BFloat16}) = (1, 8, 7)

bitwidth(::Type{T}) where T<:AbstractFloat = sum(float_bits(T))
sign_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[1]
exponent_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[2]
significand_bits(::Type{T}) where T<:AbstractFloat = float_bits(T)[3]

exponent_bias(::Type{T}) where T<:AbstractFloat = 2^(exponent_bits(T) - 1) - 1

@public bitwidth, sign_bits, exponent_bits, significand_bits

include("Microfloat.jl")
export Microfloat
@public sign_mask, exponent_mask, significand_mask
@public NonFiniteBehavior, non_finite_behavior, hasinf, hasnan, inf, nan
export IEEE, NanOnlyAllOnes, FiniteOnly

include("conversion.jl")
export OverflowPolicy, SAT, OVF
@public default_overflow_policy

include("macros.jl")
export @microfloat

# Each `@microfloat` call builds a per-type BFloat16 lookup table,
# so conversion.jl must be loaded before this point.
include("variants.jl")
export Float8_E5M2, Float8_E4M3, Float8_E3M4
export Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU
export Float6_E2M3FN, Float6_E3M2FN
export Float4_E2M1FN

include("ops.jl")
include("random.jl")

for T in (
    :Float8_E4M3FN, :Float8_E5M2, :Float8_E8M0FNU,
    :Float6_E2M3FN, :Float6_E3M2FN,
    :Float4_E2M1FN,
)
    @eval @doc """
        $($T)

    ## Properties
    - Bits: `$(sign_bits($T))` sign + `$(exponent_bits($T))` exponent + `$(significand_bits($T))` significand (`$(bitwidth($T))` total)
    - Non-finite behavior: `$(non_finite_behavior($T))`
    - Has Inf: `$(hasinf($T))`
    - Has NaN: `$(hasnan($T))`
    - Max normal: `$(Float64(floatmax($T)))`
    - Min positive: `$(Float64(floatmin($T)))`
    """ $T
end

end
