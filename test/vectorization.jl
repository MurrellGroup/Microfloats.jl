@testset "Vector aliases" begin
    @test Microfloats.Float16x2 === Microfloats.SVector{2,Float16}
    @test Microfloats.BFloat16x4 === Microfloats.SVector{4,BFloat16}

    @test Microfloats.Float8x2_E4M3FN === Microfloats.NVector{Float8_E4M3FN,2}
    @test Microfloats.Float8x4_E5M2 === Microfloats.NVector{Float8_E5M2,4}
    @test Microfloats.Float8x2_E8M0FNU === Microfloats.NVector{Float8_E8M0FNU,2}

    @test Microfloats.Float6x2_E2M3FN === Microfloats.NVector{Float6_E2M3FN,2}
    @test Microfloats.Float6x4_E3M2FN === Microfloats.NVector{Float6_E3M2FN,4}
    @test Microfloats.Float4x4_E2M1FN === Microfloats.NVector{Float4_E2M1FN,4}

    f4 = Microfloats.Float4x2_E2M1FN((Float4_E2M1FN(1), Float4_E2M1FN(2)))
    @test bitwidth(f4) == 8
    @test reinterpret(UInt8, f4) ==
          (reinterpret(UInt8, Float4_E2M1FN(1)) |
           (reinterpret(UInt8, Float4_E2M1FN(2)) << 4))

    f6 = Microfloats.Float6x4_E2M3FN((
        Float6_E2M3FN(1), Float6_E2M3FN(2),
        Float6_E2M3FN(3), Float6_E2M3FN(4),
    ))
    @test bitwidth(f6) == 24
    @test Tuple(f6) == (
        Float6_E2M3FN(1), Float6_E2M3FN(2),
        Float6_E2M3FN(3), Float6_E2M3FN(4),
    )

    f4_from_svec = Microfloats.Float4x2_E2M1FN(Microfloats.SVector{2,Float16}(1, 2))
    @test Tuple(f4_from_svec) == (Float4_E2M1FN(1), Float4_E2M1FN(2))

    f8_from_svec = Microfloats.Float8x2_E5M2(Microfloats.SVector{2,BFloat16}(BFloat16(1), BFloat16(2)))
    @test Tuple(f8_from_svec) == (Float8_E5M2(1), Float8_E5M2(2))

    f6_from_nvec = Microfloats.Float6x2_E2M3FN(f4)
    @test Tuple(f6_from_nvec) == (Float6_E2M3FN(1), Float6_E2M3FN(2))
end
