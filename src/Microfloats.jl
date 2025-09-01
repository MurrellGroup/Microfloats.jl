module Microfloats

include("float-bits.jl")

include("Microfloat.jl")
export Microfloat
export IEEE

include("microscaled/microscaled.jl")
export MX, NV
export MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1, MX_E8M0, NV_E2M1

include("conversion/conversion.jl")

include("ops.jl")

include("show.jl")

include("random.jl")

end
