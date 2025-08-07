using Microfloats
using Test

@testset "Microfloats" begin

    #include("test_standard.jl")
    #include("test_bounded.jl")

    include("verify_MX_compliance.jl")
    #include("verify_Float8s_parity.jl")

end
