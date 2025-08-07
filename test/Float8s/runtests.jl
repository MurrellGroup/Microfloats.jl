# check parity with the Float8s.jl package

using Float8s: Float8, Float8_4

const FP8_E3M4 = Microfloat(1, 3, 4)
const FP8_E4M3 = Microfloat(1, 4, 3)

@testset "Float8s.jl parity" begin

    @testset "E3M4" begin

        @testset for i in 0x00:0xfe
            @test FP8_E3M4(Float32(reinterpret(Float8, i))) ≡ reinterpret(FP8_E3M4, i)

            @test Float8(Float32(reinterpret(FP8_E3M4, i))) ≡ reinterpret(Float8, i)

            @test Float32(reinterpret(Float8, i)) ≡
                  Float32(reinterpret(FP8_E3M4, i))

            @test Float32(Float8(Float32(reinterpret(Float8, i)))) ≡
                  Float32(FP8_E3M4(Float32(reinterpret(FP8_E3M4, i))))
        end

    end

    @testset "E4M3" begin

        @testset for i in 0x00:0xfe
            @test FP8_E4M3(Float32(reinterpret(Float8_4, i))) ≡ reinterpret(FP8_E4M3, i)

            @test Float8_4(Float32(reinterpret(FP8_E4M3, i))) ≡ reinterpret(Float8_4, i)

            @test Float32(reinterpret(Float8_4, i)) ≡
                  Float32(reinterpret(FP8_E4M3, i))

            @test Float32(Float8_4(Float32(reinterpret(Float8_4, i)))) ≡
                  Float32(FP8_E4M3(Float32(reinterpret(FP8_E4M3, i))))
        end

    end

end
