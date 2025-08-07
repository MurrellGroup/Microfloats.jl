using Microfloats
using Test

a ≡ b = isnan(a) || isnan(b) ? true : a == b

@testset "Microfloats" begin

    include("MX_compliance.jl")
    include("MX_properties.jl")
    include("Float8s_parity.jl")

end
