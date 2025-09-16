using Microfloats
using Test

using BitPacking

a ≡ b = isnan(a) || isnan(b) ? true : a == b

@testset "Microfloats" begin

    include("Microfloat.jl")
    include("overflow.jl")

    include("Float8s/runtests.jl")
    include("MX/runtests.jl")

    @testset "BitPackingExt" begin
        x = randn(Float4_E2M1, 16)
        y = bitpacked(x)
        @test y isa BitPackedArray
        @test x == y
        @test x == bitunpacked(y)
    end

end
