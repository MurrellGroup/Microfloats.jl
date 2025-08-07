# Section 5.3
# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

@testset "MX specification compliance" begin

    @testset "Element Data Types" begin
    
        @testset "FP8" begin

            @testset "E4M3" begin
                E4M3 = Microfloat(1, 4, 3; variant=:MX)

                @test Microfloats.bias(E4M3) == 7

                @test isfinite(reinterpret(E4M3, 0b0_1111_000))
                @test isfinite(reinterpret(E4M3, 0b1_1111_000))

                for i in 0b001:0b110
                    @test isfinite(reinterpret(E4M3, 0b0_1111_000 | i))
                    @test isfinite(reinterpret(E4M3, 0b1_1111_000 | i))
                end

                @test isnan(reinterpret(E4M3, 0b0_1111_111))
                @test isnan(reinterpret(E4M3, 0b1_1111_111))

                @test iszero(reinterpret(E4M3, 0b0_0000_000))
                @test iszero(reinterpret(E4M3, 0b1_0000_000))

                @test reinterpret(E4M3, 0b0_1111_110) == 2^8 * 1.75
                @test reinterpret(E4M3, 0b1_1111_110) == -2^8 * 1.75

                @test reinterpret(E4M3, 0b0_0001_000) == 2^-6
                @test reinterpret(E4M3, 0b1_0001_000) == -2^-6

                @test reinterpret(E4M3, 0b0_0000_111) == 2^-6 * 0.875
                @test reinterpret(E4M3, 0b1_0000_111) == -2^-6 * 0.875

                @test reinterpret(E4M3, 0b0_0000_001) == 2^-9
                @test reinterpret(E4M3, 0b1_0000_001) == -2^-9
            end

            @testset "E5M2" begin
                E5M2 = Microfloat(1, 5, 2)

                @test Microfloats.bias(E5M2) == 15

                @test reinterpret(UInt8, E5M2(Inf)) == 0b0_11111_00
                @test reinterpret(UInt8, E5M2(-Inf)) == 0b1_11111_00

                for i in 0b01:0b11
                    @test isnan(reinterpret(E5M2, 0b0_11111_00 | i))
                    @test isnan(reinterpret(E5M2, 0b1_11111_00 | i))
                end

                @test iszero(reinterpret(E5M2, 0b0_00000_00))
                @test iszero(reinterpret(E5M2, 0b1_00000_00))

                @test reinterpret(E5M2, 0b0_11110_11) == 2^15 * 1.75
                @test reinterpret(E5M2, 0b1_11110_11) == -2^15 * 1.75
                @test nextfloat(reinterpret(E5M2, 0b0_11110_11)) == Inf
                @test prevfloat(reinterpret(E5M2, 0b1_11110_11)) == -Inf

                @test reinterpret(E5M2, 0b0_00001_00) == 2^-14
                @test reinterpret(E5M2, 0b1_00001_00) == -2^-14

                @test reinterpret(E5M2, 0b0_00000_11) == 2^-14 * 0.75
                @test reinterpret(E5M2, 0b1_00000_11) == -2^-14 * 0.75
                @test reinterpret(E5M2, 0b0_00000_01) == 2^-16
                @test reinterpret(E5M2, 0b1_00000_01) == -2^-16
            end

        end

        @testset "FP6" begin

            @testset "E2M3" begin
                E2M3 = Microfloat(1, 2, 3; variant=:MX)

                @test Microfloats.bias(E2M3) == 1

                @test isfinite(reinterpret(E2M3, 0b0_11_000_00))
                @test isfinite(reinterpret(E2M3, 0b1_11_000_00))

                for i in 0b001:0b111
                    @test isfinite(reinterpret(E2M3, 0b0_11_000_00 | i << 2))
                    @test isfinite(reinterpret(E2M3, 0b1_11_000_00 | i << 2))
                end

                @test iszero(reinterpret(E2M3, 0b0_00_000_00))
                @test iszero(reinterpret(E2M3, 0b1_00_000_00))

                @test reinterpret(E2M3, 0b0_11_111_00) == 2^2 * 1.875
                @test reinterpret(E2M3, 0b1_11_111_00) == -2^2 * 1.875

                @test reinterpret(E2M3, 0b0_01_000_00) == 2^0 * 1.0
                @test reinterpret(E2M3, 0b1_01_000_00) == -2^0 * 1.0

                @test reinterpret(E2M3, 0b0_00_111_00) == 2^0 * 0.875
                @test reinterpret(E2M3, 0b1_00_111_00) == -2^0 * 0.875

                @test reinterpret(E2M3, 0b0_00_001_00) == 2^0 * 0.125
                @test reinterpret(E2M3, 0b1_00_001_00) == -2^0 * 0.125

            end

            @testset "E3M2" begin
                E3M2 = Microfloat(1, 3, 2; variant=:MX)

                @test Microfloats.bias(E3M2) == 3

                @test isfinite(reinterpret(E3M2, 0b0_111_00_00))
                @test isfinite(reinterpret(E3M2, 0b1_111_00_00))

                for i in 0b01:0b11
                    @test isfinite(reinterpret(E3M2, 0b0_111_00_00 | i << 2))
                    @test isfinite(reinterpret(E3M2, 0b1_111_00_00 | i << 2))
                end

                @test iszero(reinterpret(E3M2, 0b0_000_00_00))
                @test iszero(reinterpret(E3M2, 0b1_000_00_00))

                @test reinterpret(E3M2, 0b0_111_11_00) == 2^4 * 1.75
                @test reinterpret(E3M2, 0b1_111_11_00) == -2^4 * 1.75

                @test reinterpret(E3M2, 0b0_001_00_00) == 2^-2 * 1.0
                @test reinterpret(E3M2, 0b1_001_00_00) == -2^-2 * 1.0

                @test reinterpret(E3M2, 0b0_000_11_00) == 2^-2 * 0.75
                @test reinterpret(E3M2, 0b1_000_11_00) == -2^-2 * 0.75

                @test reinterpret(E3M2, 0b0_000_01_00) == 2^-2 * 0.25
                @test reinterpret(E3M2, 0b1_000_01_00) == -2^-2 * 0.25
            end

        end

        @testset "FP4" begin

            @testset "E2M1" begin
                E2M1 = Microfloat(1, 2, 1; variant=:MX)

                @test Microfloats.bias(E2M1) == 1

                @test isfinite(reinterpret(E2M1, 0b0_11_0_0000))
                @test isfinite(reinterpret(E2M1, 0b1_11_0_0000))

                @test isfinite(reinterpret(E2M1, 0b0_11_1_0000))
                @test isfinite(reinterpret(E2M1, 0b1_11_1_0000))

                @test iszero(reinterpret(E2M1, 0b0_00_0_0000))
                @test iszero(reinterpret(E2M1, 0b1_00_0_0000))

                @test reinterpret(E2M1, 0b0_11_1_0000) == 2^2 * 1.5
                @test reinterpret(E2M1, 0b1_11_1_0000) == -2^2 * 1.5

                @test reinterpret(E2M1, 0b0_01_0_0000) == 2^0 * 1.0
                @test reinterpret(E2M1, 0b1_01_0_0000) == -2^0 * 1.0

                @test reinterpret(E2M1, 0b0_00_1_0000) == 2^0 * 0.5
                @test reinterpret(E2M1, 0b1_00_1_0000) == -2^0 * 0.5
            end

        end

    end

    @testset "Scale Data Types" begin

        # arithmetic not yet supported for unsigned microfloats
        @testset "E8M0" begin
            E8M0 = Microfloat(0, 8, 0; variant=:MX)

            @test Microfloats.bias(E8M0) == 127

            #@test floatmax(E8M0) == floatmax(Float32) / 2

            @test !isfinite(reinterpret(E8M0, 0b11111111))

            @test isnan(reinterpret(E8M0, 0b11111111))

            @test iszero(reinterpret(E8M0, 0b00000000))
        end

    end

end