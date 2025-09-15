module Microfloats

import BFloat16s: BFloat16
export BFloat16

include("Microfloat.jl")
include("conversion.jl")
include("show.jl")
include("ops.jl")
include("random.jl")

"""
    Microfloat(S, E, M, V=IEEE)

Create a new `Microfloat` type with `S` sign bits, `E` exponent bits, and `M` mantissa bits.

This "type constructor" ensures that the resulting type is legal.

The `V` argument can be set to `MX` to create a Microscaling Format (MX) type.
"""
function Microfloat(S::Int, E::Int, M::Int, V::Type{<:FloatVariant}=FloatVariant)
    S in (0, 1) || throw(ArgumentError("sign bit must be 0 or 1"))
    E >= 1 || throw(ArgumentError("number of exponent bits must be non-negative"))
    M >= 0 || throw(ArgumentError("number of significand bits must be non-negative"))
    0 < S + E + M <= 8 || throw(ArgumentError("total number of bits must be between 1 and 8"))
    return Microfloat{S,E,M,V}
end

const Float8_E3M4 = Microfloat(1,3,4)
const Float8_E4M3 = Microfloat(1,4,3)
const Float8_E5M2 = Microfloat(1,5,2)
const Float6_E2M3 = Microfloat(1,2,3)
const Float6_E3M2 = Microfloat(1,3,2)
const Float4_E2M1 = Microfloat(1,2,1)

export Microfloat
export OVF, SAT
export Float8_E3M4, Float8_E4M3, Float8_E5M2
export Float6_E2M3, Float6_E3M2
export Float4_E2M1

include("MX.jl")

end
