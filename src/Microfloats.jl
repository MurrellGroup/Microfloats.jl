module Microfloats

include("float-bits.jl")

include("Microfloat.jl")
export Microfloat
export Float8_E3M4, Float8_E4M3, Float8_E5M2
export Float6_E2M3, Float6_E3M2
export Float4_E2M1

include("microscaled/microscaled.jl")
export MX_E5M2, MX_E4M3
export MX_E3M2, MX_E2M3
export MX_E2M1
export MX_E8M0

include("conversion/conversion.jl")

include("ops.jl")

include("show.jl")

include("random.jl")

end
