@testset "Overflow: IEEE types" begin
    # Default: OVF → Inf for finite overflow, NaN input → NaN
    @testset "$T" for T in (Float8_E3M4, Float8_E5M2)
        @test default_overflow_policy(T) === OVF

        @test isnan(T(NaN))
        @test T(+Inf) == +inf(T)
        @test T(-Inf) == -inf(T)

        big = nextfloat(BFloat16(floatmax(T)))
        @test T(+big) == +inf(T)       # default OVF
        @test T(-big) == -inf(T)
        @test T(+big, SAT) == +floatmax(T)
        @test T(-big, SAT) == -floatmax(T)
    end
end

@testset "Overflow: NanOnlyAllOnes types" begin
    # Default: SAT → floatmax for finite overflow (matches PyTorch/Triton/Quartet)
    @testset "$T" for T in (Float8_E4M3FN,)
        @test default_overflow_policy(T) === SAT

        @test isnan(T(NaN))
        @test T(+Inf) == +floatmax(T)
        @test T(-Inf) == -floatmax(T)

        big = nextfloat(BFloat16(floatmax(T)))
        @test T(+big) == +floatmax(T)             # default SAT
        @test T(-big) == -floatmax(T)
        @test isnan(T(+big, OVF))
        @test isnan(T(-big, OVF))
    end

    # Unsigned NanOnlyAllOnes (E8M0FNU): negative input throws, large positive saturates
    T = Float8_E8M0FNU
    @test default_overflow_policy(T) === SAT
    @test T(+Inf) == floatmax(T)
    @test_throws DomainError T(-Inf)
    big = nextfloat(BFloat16(floatmax(T)))
    @test T(big) == floatmax(T)
    @test isnan(T(big, OVF))
    @test isnan(T(NaN))
end

@testset "Overflow: FiniteOnly types" begin
    # Default: SAT. NaN input always throws (no sentinel). OVF also throws on overflow.
    @testset "$T" for T in (Float4_E2M1FN, Float6_E2M3FN, Float6_E3M2FN)
        @test default_overflow_policy(T) === SAT

        @test_throws DomainError T(NaN)
        @test_throws DomainError T(NaN, OVF)
        @test_throws DomainError T(NaN, SAT)

        @test T(+Inf) == +floatmax(T)
        @test T(-Inf) == -floatmax(T)
        @test_throws DomainError T(+Inf, OVF)
        @test_throws DomainError T(-Inf, OVF)

        big = nextfloat(BFloat16(floatmax(T)))
        @test T(+big) == +floatmax(T)
        @test T(-big) == -floatmax(T)
        @test_throws DomainError T(+big, OVF)
        @test_throws DomainError T(-big, OVF)
    end

    # Specific: Float4_E2M1FN (NVFP4 value type) saturates at ±6
    @test Float32(Float4_E2M1FN(6.0)) == 6.0
    @test Float32(Float4_E2M1FN(7.0)) == 6.0   # SAT
    @test_throws DomainError Float4_E2M1FN(7.0, OVF)
end
