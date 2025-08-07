module Microfloats

include("Microfloat.jl")
export Microfloat
export SignedMicrofloat, UnsignedMicrofloat
export SignedStandardMicrofloat, UnsignedStandardMicrofloat
export SignedBoundedMicrofloat, UnsignedBoundedMicrofloat

include("MX.jl")

include("conversion/conversion.jl")

include("ops.jl")

include("show.jl")


end
