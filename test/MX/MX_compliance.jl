# Section 5.3
# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

@testset "MX specification compliance" begin

    @testset "Element Data Types" begin
    
        @testset "FP8" begin

            @testset "E4M3" begin
                @test Microfloats.bias(MX_E4M3) == 7

                @test isfinite(reinterpret(MX_E4M3, 0b0_1111_000))
                @test isfinite(reinterpret(MX_E4M3, 0b1_1111_000))

                for i in 0b001:0b110
                    @test isfinite(reinterpret(MX_E4M3, 0b0_1111_000 | i))
                    @test isfinite(reinterpret(MX_E4M3, 0b1_1111_000 | i))
                end

                @test isnan(reinterpret(MX_E4M3, 0b0_1111_111))
                @test isnan(reinterpret(MX_E4M3, 0b1_1111_111))

                @test iszero(reinterpret(MX_E4M3, 0b0_0000_000))
                @test iszero(reinterpret(MX_E4M3, 0b1_0000_000))

                @test reinterpret(MX_E4M3, 0b0_1111_110) == 2^8 * 1.75 == floatmax(MX_E4M3)
                @test reinterpret(MX_E4M3, 0b1_1111_110) == -2^8 * 1.75 == -floatmax(MX_E4M3)

                @test reinterpret(MX_E4M3, 0b0_0001_000) == 2^-6 == floatmin(MX_E4M3)
                @test reinterpret(MX_E4M3, 0b1_0001_000) == -2^-6 == -floatmin(MX_E4M3)

                @test reinterpret(MX_E4M3, 0b0_0000_111) == 2^-6 * 0.875 == prevfloat(floatmin(MX_E4M3))
                @test reinterpret(MX_E4M3, 0b1_0000_111) == -2^-6 * 0.875 == -prevfloat(floatmin(MX_E4M3))

                @test reinterpret(MX_E4M3, 0b0_0000_001) == 2^-9 == nextfloat(zero(MX_E4M3))
                @test reinterpret(MX_E4M3, 0b1_0000_001) == -2^-9 == -nextfloat(zero(MX_E4M3))
            end

            @testset "E5M2" begin
                @test Microfloats.bias(MX_E5M2) == 15

                @test reinterpret(UInt8, MX_E5M2(Inf)) == 0b0_11111_00
                @test reinterpret(UInt8, MX_E5M2(-Inf)) == 0b1_11111_00

                for i in 0b01:0b11
                    @test isnan(reinterpret(MX_E5M2, 0b0_11111_00 | i))
                    @test isnan(reinterpret(MX_E5M2, 0b1_11111_00 | i))
                end

                @test iszero(reinterpret(MX_E5M2, 0b0_00000_00))
                @test iszero(reinterpret(MX_E5M2, 0b1_00000_00))

                @test reinterpret(MX_E5M2, 0b0_11110_11) == 2^15 * 1.75 == floatmax(MX_E5M2)
                @test reinterpret(MX_E5M2, 0b1_11110_11) == -2^15 * 1.75 == -floatmax(MX_E5M2)
                @test nextfloat(reinterpret(MX_E5M2, 0b0_11110_11)) == Inf
                @test prevfloat(reinterpret(MX_E5M2, 0b1_11110_11)) == -Inf

                @test reinterpret(MX_E5M2, 0b0_00001_00) == 2^-14 == floatmin(MX_E5M2)
                @test reinterpret(MX_E5M2, 0b1_00001_00) == -2^-14 == -floatmin(MX_E5M2)

                @test reinterpret(MX_E5M2, 0b0_00000_11) == 2^-14 * 0.75 == prevfloat(floatmin(MX_E5M2))
                @test reinterpret(MX_E5M2, 0b1_00000_11) == -2^-14 * 0.75 == -prevfloat(floatmin(MX_E5M2))

                @test reinterpret(MX_E5M2, 0b0_00000_01) == 2^-16 == nextfloat(zero(MX_E5M2))
                @test reinterpret(MX_E5M2, 0b1_00000_01) == -2^-16 == -nextfloat(zero(MX_E5M2))
            end

        end

        @testset "FP6" begin

            @testset "E2M3" begin
                @test Microfloats.bias(MX_E2M3) == 1

                @test isfinite(reinterpret(MX_E2M3, 0b0_11_000))
                @test isfinite(reinterpret(MX_E2M3, 0b1_11_000))

                for i in 0b001:0b111
                    @test isfinite(reinterpret(MX_E2M3, 0b0_11_000 | i << 2))
                    @test isfinite(reinterpret(MX_E2M3, 0b1_11_000 | i << 2))
                end

                @test iszero(reinterpret(MX_E2M3, 0b0_00_000))
                @test iszero(reinterpret(MX_E2M3, 0b1_00_000))

                @test reinterpret(MX_E2M3, 0b0_11_111) == 2^2 * 1.875 == floatmax(MX_E2M3)
                @test reinterpret(MX_E2M3, 0b1_11_111) == -2^2 * 1.875 == -floatmax(MX_E2M3)

                @test reinterpret(MX_E2M3, 0b0_01_000) == 2^0 * 1.0 == floatmin(MX_E2M3)
                @test reinterpret(MX_E2M3, 0b1_01_000) == -2^0 * 1.0 == -floatmin(MX_E2M3)

                @test reinterpret(MX_E2M3, 0b0_00_111) == 2^0 * 0.875 == prevfloat(floatmin(MX_E2M3))
                @test reinterpret(MX_E2M3, 0b1_00_111) == -2^0 * 0.875 == -prevfloat(floatmin(MX_E2M3))

                @test reinterpret(MX_E2M3, 0b0_00_001) == 2^0 * 0.125 == nextfloat(zero(MX_E2M3))
                @test reinterpret(MX_E2M3, 0b1_00_001) == -2^0 * 0.125 == -nextfloat(zero(MX_E2M3))
            end

            @testset "E3M2" begin
                @test Microfloats.bias(MX_E3M2) == 3

                @test isfinite(reinterpret(MX_E3M2, 0b0_111_00))
                @test isfinite(reinterpret(MX_E3M2, 0b1_111_00))

                for i in 0b01:0b11
                    @test isfinite(reinterpret(MX_E3M2, 0b0_111_00 | i << 2))
                    @test isfinite(reinterpret(MX_E3M2, 0b1_111_00 | i << 2))
                end

                @test iszero(reinterpret(MX_E3M2, 0b0_000_00))
                @test iszero(reinterpret(MX_E3M2, 0b1_000_00))

                @test reinterpret(MX_E3M2, 0b0_111_11) == 2^4 * 1.75 == floatmax(MX_E3M2)
                @test reinterpret(MX_E3M2, 0b1_111_11) == -2^4 * 1.75 == -floatmax(MX_E3M2)

                @test reinterpret(MX_E3M2, 0b0_001_00) == 2^-2 * 1.0 == floatmin(MX_E3M2)
                @test reinterpret(MX_E3M2, 0b1_001_00) == -2^-2 * 1.0 == -floatmin(MX_E3M2)

                @test reinterpret(MX_E3M2, 0b0_000_11) == 2^-2 * 0.75 == prevfloat(floatmin(MX_E3M2))
                @test reinterpret(MX_E3M2, 0b1_000_11) == -2^-2 * 0.75 == -prevfloat(floatmin(MX_E3M2))

                @test reinterpret(MX_E3M2, 0b0_000_01) == 2^-2 * 0.25 == nextfloat(zero(MX_E3M2))
                @test reinterpret(MX_E3M2, 0b1_000_01) == -2^-2 * 0.25 == -nextfloat(zero(MX_E3M2))
            end

        end

        @testset "FP4" begin

            @testset "E2M1" begin
                @test Microfloats.bias(MX_E2M1) == 1

                @test isfinite(reinterpret(MX_E2M1, 0b0_11_0))
                @test isfinite(reinterpret(MX_E2M1, 0b1_11_0))

                @test isfinite(reinterpret(MX_E2M1, 0b0_11_1))
                @test isfinite(reinterpret(MX_E2M1, 0b1_11_1))

                @test iszero(reinterpret(MX_E2M1, 0b0_00_0))
                @test iszero(reinterpret(MX_E2M1, 0b1_00_0))

                @test reinterpret(MX_E2M1, 0b0_11_1) == 2^2 * 1.5 == floatmax(MX_E2M1)
                @test reinterpret(MX_E2M1, 0b1_11_1) == -2^2 * 1.5 == -floatmax(MX_E2M1)

                @test reinterpret(MX_E2M1, 0b0_01_0) == 2^0 * 1.0 == floatmin(MX_E2M1)
                @test reinterpret(MX_E2M1, 0b1_01_0) == -2^0 * 1.0 == -floatmin(MX_E2M1)

                @test reinterpret(MX_E2M1, 0b0_00_1) == 2^0 * 0.5 == nextfloat(zero(MX_E2M1))
                @test reinterpret(MX_E2M1, 0b1_00_1) == -2^0 * 0.5 == -nextfloat(zero(MX_E2M1))
            end

        end

    end

    @testset "Scale Data Types" begin

        # arithmetic not yet supported for unsigned microfloats
        @testset "E8M0" begin
            @test Microfloats.bias(MX_E8M0) == 127

            #@test floatmax(E8M0) == floatmax(Float32) / 2

            @test !isfinite(reinterpret(MX_E8M0, 0b11111111))

            @test isnan(reinterpret(MX_E8M0, 0b11111111))

            @test iszero(reinterpret(MX_E8M0, 0b00000000))
        end

    end

end