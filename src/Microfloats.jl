module Microfloats

import BFloat16s: BFloat16
export BFloat16

include("Microfloat.jl")
export Microfloat

include("conversion.jl")
export OVF, SAT

include("IEEE.jl")
export Float8_E5M2
export Float8_E4M3
export Float8_E3M4
export Float6_E3M2
export Float6_E2M3
export Float4_E2M1

include("MX.jl")
export MX_E5M2
export MX_E4M3
export MX_E3M2
export MX_E2M3
export MX_E2M1
export MX_E8M0

include("ops.jl")
include("show.jl")
include("random.jl")

"""
    Microfloat(S, E, M, V=IEEE)

Create a new `Microfloat` type with `S` sign bits, `E` exponent bits, and `M` mantissa bits.

This "type constructor" ensures that the resulting type is legal.

The `V` argument can be set to `MX` to create a Microscaling Format (MX) type.
"""
function Microfloat(S::Int, E::Int, M::Int, V::Type{<:Finite}=IEEE)
    S in (0, 1) || throw(ArgumentError("sign bit must be 0 or 1"))
    E >= 1 || throw(ArgumentError("number of exponent bits must be non-negative"))
    M >= 0 || throw(ArgumentError("number of significand bits must be non-negative"))
    0 < S + E + M <= 8 || throw(ArgumentError("total number of bits must be between 1 and 8"))
    return Microfloat{S,E,M,V}
end

end
