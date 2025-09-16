module Microfloats

import BFloat16s: BFloat16
export BFloat16

include("Microfloat.jl")
export Finite
export Microfloat

include("variants/Finite.jl")
export Finite

include("variants/IEEE_754_like.jl")
export IEEE_754_like
export Float8_E5M2
export Float8_E4M3
export Float8_E3M4
export Float6_E3M2
export Float6_E2M3
export Float4_E2M1

include("variants/MX.jl")
export MX
export MX_E5M2
export MX_E4M3
export MX_E3M2
export MX_E2M3
export MX_E2M1
export MX_E8M0

# something weird happens when the @generated BFloat16
# method is put before the other includes.
# presumably precompilation is being excessively greedy
include("conversion.jl")
export OVF, SAT

include("ops.jl")
include("show.jl")
include("random.jl")

"""
    Microfloat{S,E,M,V}

A `Microfloat` type has `S` sign bits (between 0 and 1),
`E` exponent bits (between 1 and 8), and `M` significand bits (between 0 and 7).
"""
Microfloat

"""
    OVF

Overflow policy for converting from BFloat16 to a `Microfloat`.
"""
OVF

for T in (
    :Float8_E5M2, :Float8_E4M3, :Float8_E3M4, :Float6_E3M2, :Float6_E2M3, :Float4_E2M1,
    :MX_E5M2, :MX_E4M3, :MX_E3M2, :MX_E2M3, :MX_E2M1, :MX_E8M0,
)
    @eval begin
        @doc """
            $($T)

        ## Properties
        - Bits: $(sign_bits($T)) sign + $(exponent_bits($T)) exponent + $(significand_bits($T)) significand ($(total_bits($T)) total)
        - Variant: $(variant($T))
        - Has Inf: $(hasinf($T))
        - Has NaN: $(hasnan($T))
        - Max normal: $(Float64(floatmax($T)))
        - Min normal: $(Float64(floatmin($T)))
        - Max subnormal: $(Float64(prevfloat(floatmin($T))))
        - Min subnormal: $(Float64(nextfloat(zero($T))))
        """
        $T
    end
end

end
