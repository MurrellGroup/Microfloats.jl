using Microfloats
using Test

a ≡ b = isnan(a) || isnan(b) ? true : a == b

uint8(x) = reinterpret(UInt8, x)

@testset "Microfloats" begin

    include("Float8s/runtests.jl")
    include("IEEE_properties.jl")
    include("MX/runtests.jl")

end
