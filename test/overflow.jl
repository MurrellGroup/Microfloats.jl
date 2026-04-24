@testset "Overflow: IEEE types (OVF)" begin
    # overflow=OVF default: finite overflow → ±Inf, NaN input → NaN.
    @testset "$T" for T in (Float8_E3M4, Float8_E5M2)
        @test overflow_policy(T) === OVF

        @test isnan(T(NaN))
        @test T(+Inf) == +inf(T)
        @test T(-Inf) == -inf(T)

        big = nextfloat(BFloat16(floatmax(T)))
        @test T(+big) == +inf(T)
        @test T(-big) == -inf(T)
    end
end

@testset "Overflow: NanOnlyAllOnes types (OVF)" begin
    # Default for NanOnlyAllOnes: OVF. Overflow → NaN. Matches cutile-python
    # and the OCP strict reading. PyTorch-style saturation requires a
    # twin type declared with `overflow=SAT`.
    @testset "Float8_E4M3FN" begin
        T = Float8_E4M3FN
        @test overflow_policy(T) === OVF

        @test isnan(T(NaN))
        @test isnan(T(+Inf))
        @test isnan(T(-Inf))

        big = nextfloat(BFloat16(floatmax(T)))
        @test isnan(T(+big))
        @test isnan(T(-big))
    end

    @testset "Float8_E8M0FNU" begin
        # Unsigned NanOnlyAllOnes scale type; negative input throws regardless.
        T = Float8_E8M0FNU
        @test overflow_policy(T) === OVF

        @test isnan(T(NaN))
        @test isnan(T(+Inf))
        @test_throws DomainError T(-Inf)

        big = nextfloat(BFloat16(floatmax(T)))
        @test isnan(T(big))
    end
end

@testset "Overflow: FiniteOnly types (SAT)" begin
    # overflow=SAT forced — no sentinel encoding exists. NaN input throws.
    @testset "$T" for T in (Float4_E2M1FN, Float6_E2M3FN, Float6_E3M2FN)
        @test overflow_policy(T) === SAT

        @test_throws DomainError T(NaN)

        @test T(+Inf) == +floatmax(T)
        @test T(-Inf) == -floatmax(T)

        big = nextfloat(BFloat16(floatmax(T)))
        @test T(+big) == +floatmax(T)
        @test T(-big) == -floatmax(T)
    end

    # Specific: Float4_E2M1FN (NVFP4 value type) saturates at ±6
    @test Float32(Float4_E2M1FN(6.0)) == 6.0
    @test Float32(Float4_E2M1FN(7.0)) == 6.0
end

@testset "Alternate policy via twin type + reinterpret" begin
    # _E4M3FN_SAT is declared at top of runtests.jl: same bit layout as
    # Float8_E4M3FN but with overflow=SAT (PyTorch/Triton convention).
    # Shows the documented escape hatch for the non-default policy.
    @test overflow_policy(_E4M3FN_SAT) === SAT
    big = nextfloat(BFloat16(floatmax(_E4M3FN_SAT)))
    @test _E4M3FN_SAT(big) == floatmax(_E4M3FN_SAT)            # SAT: overflow → floatmax
    @test isnan(Float8_E4M3FN(big))                            # OVF: overflow → NaN
    # Bit layout is identical, so reinterpret is a free relabel.
    x = Float8_E4M3FN(1.0)
    @test reinterpret(UInt8, reinterpret(_E4M3FN_SAT, x)) == reinterpret(UInt8, x)
end
