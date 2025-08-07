using Microfloats
using Test

a ≡ b = isnan(a) || isnan(b) ? true : a == b

@testset "Microfloats" begin

    include("Float8s/runtests.jl")
    include("MX/runtests.jl")

end
