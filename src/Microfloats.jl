module Microfloats

using Republic

@republic import BitPacking: bitwidth
@republic import BFloat16s: BFloat16

include("utils.jl")
@public sign_bits, exponent_bits, significand_bits

include("NonFiniteBehavior.jl")
@public IEEE, NanOnlyAllOnes, FiniteOnly

include("Microfloat.jl")
export Microfloat
@public hasinf, hasnan
@public non_finite_behavior

include("conversion.jl")
@public overflow_policy
@public SAT, OVF

include("macro.jl")
export @microfloat

include("variants.jl")
export Float8_E5M2, Float8_E4M3, Float8_E3M4
export Float8_E4M3FN, Float8_E8M0FNU
export Float6_E2M3FN, Float6_E3M2FN
export Float4_E2M1FN

include("vectorization.jl")
@public Float16x2, Float16x4
@public BFloat16x2, BFloat16x4
@public Float8x2_E4M3FN, Float8x4_E4M3FN
@public Float8x2_E5M2, Float8x4_E5M2
@public Float8x2_E8M0FNU, Float8x4_E8M0FNU
@public Float6x2_E2M3FN, Float6x4_E2M3FN
@public Float6x2_E3M2FN, Float6x4_E3M2FN
@public Float4x2_E2M1FN, Float4x4_E2M1FN

include("ops.jl")
include("random.jl")

end
