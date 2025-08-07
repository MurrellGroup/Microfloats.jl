module Microfloats

include("float-bits.jl")

include("Microfloat.jl")
export Microfloat
export SignedMicrofloat, UnsignedMicrofloat

include("MX/MX.jl")

include("conversion/conversion.jl")

include("ops.jl")

include("show.jl")


end
