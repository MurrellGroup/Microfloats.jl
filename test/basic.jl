@testset "Microfloat generic properties" begin
    @testset "$T" for T in TYPES
        @test hash(one(T)) == hash(1)
        @test precision(T) == Microfloats.significand_bits(T) + 1
        @test floatmax(T) > zero(T)
        @test isfinite(floatmax(T))
        @test Microfloats.overflow_policy(T) <: Union{Microfloats.OVF, Microfloats.SAT}

        @test signbit(zero(T)) == false

        if hasinf(T)
            @test typemax(T) == inf(T)
            @test floatmax(T) == prevfloat(inf(T))
        else
            @test typemax(T) == floatmax(T)
        end

        if sign_bits(T) == 1
            @test typemin(T) == (hasinf(T) ? -inf(T) : -floatmax(T))
        else
            @test typemin(T) == zero(T)
        end

        # nextfloat/prevfloat consistency
        @test nextfloat(zero(T)) > zero(T)
        sign_bits(T) == 1 && @test prevfloat(zero(T)) < zero(T)

        # round-trip: every canonical bit pattern goes out to BFloat16 and back
        used_mask = Microfloats.sign_mask(T) | Microfloats.exponent_mask(T) | Microfloats.significand_mask(T)
        @testset "round-trip" for u in 0x00:0xff
            (u & ~used_mask) != 0x00 && continue
            x = reinterpret(T, u)
            isnan(x) && continue
            @test T(BFloat16(x)) ≡ x
            @test T(Float32(x)) ≡ x
        end
    end

    @test widen(Float8_E4M3) == BFloat16
    @test string(Float8_E4M3(1.0)) == "Float8_E4M3(1.0)"
end

@testset "@microfloat" begin
    @test_throws "abc" @eval @microfloat Name abc=1
end

@testset "NonFiniteBehavior trait" begin
    @test hasinf(IEEE) && hasnan(IEEE)
    @test !hasinf(NanOnlyAllOnes) && hasnan(NanOnlyAllOnes)
    @test !hasinf(FiniteOnly) && !hasnan(FiniteOnly)

    @test non_finite_behavior(Float8_E5M2)      === IEEE
    @test non_finite_behavior(Float8_E4M3FN)    === NanOnlyAllOnes
    @test non_finite_behavior(Float8_E8M0FNU)   === NanOnlyAllOnes
    @test non_finite_behavior(Float4_E2M1FN)    === FiniteOnly

    # Forgetting `non_finite_behavior` on a custom type errors loudly.
    primitive type _BadFloat <: Microfloats.Microfloat{1,2,1} 8 end
    @test_throws ErrorException Microfloats.non_finite_behavior(_BadFloat)
end

@testset "overflow_policy trait" begin
    @test Microfloats.overflow_policy(Float8_E4M3FN) === OVF
    @test Microfloats.overflow_policy(_E4M3FN_SAT)   === SAT

    # Forgetting `overflow_policy` on a custom type errors loudly.
    primitive type _BadFloatOvf <: Microfloats.Microfloat{1,2,1} 8 end
    @test_throws ErrorException Microfloats.overflow_policy(_BadFloatOvf)
end

@testset "IEEE types: Inf and NaN encodings" begin
    @testset "$T" for T in (Float8_E3M4, Float8_E5M2)
        @test isinf(inf(T))
        @test !isnan(inf(T))
        @test isnan(nan(T))
        @test !isinf(nan(T))
        @test isinf(T(Inf))
        @test isinf(T(-Inf))
        @test isnan(T(NaN))
    end
end

@testset "NanOnlyAllOnes types: no Inf" begin
    @testset "$T" for T in (Float8_E4M3FN, Float8_E8M0FNU)
        for u in 0x00:0xff
            @test !isinf(reinterpret(T, u))
        end
        @test_throws DomainError inf(T)
        @test isnan(nan(T))
    end
end

@testset "FiniteOnly types: no Inf or NaN" begin
    @testset "$T" for T in (Float4_E2M1FN, Float6_E2M3FN, Float6_E3M2FN)
        for u in 0x00:0xff
            x = reinterpret(T, u)
            @test !isinf(x)
            @test !isnan(x)
        end
        @test_throws DomainError inf(T)
        @test_throws DomainError nan(T)
    end
end

@testset "Signed zero preservation" begin
    for T in SIGNED_TYPES
        @testset "$T" begin
            nz = T(-0.0)
            @test iszero(nz)
            @test signbit(nz)
            @test Float32(nz) === -0.0f0
            @test nz == zero(T)
            @test signbit(nz) != signbit(zero(T))
        end
    end
end

@testset "Unsigned microfloats" begin
    @test_throws DomainError Float8_E8M0FNU(-1.0)
    @test_throws DomainError Float8_E8M0FNU(-0.0)
    @test_throws DomainError -one(Float8_E8M0FNU)
    @test_throws ArgumentError randn(Float8_E8M0FNU)

    # Positive round-trip through BF16 is lossless for powers of 2 (the only E8M0 values)
    @test Float32(Float8_E8M0FNU(1.0))  == 1.0
    @test Float32(Float8_E8M0FNU(2.0))  == 2.0
    @test Float32(Float8_E8M0FNU(0.5))  == 0.5
    @test Float32(reinterpret(Float8_E8M0FNU, 0x00)) == 2f0^-127
    @test !iszero(reinterpret(Float8_E8M0FNU, 0x00))
end

@testset "rand / randn" begin
    rng = MersenneTwister(123)
    @testset "$T rand" for T in TYPES
        xs = rand(rng, T, 1000)
        @test all(isfinite, xs)
        @test any(x -> x != zero(T), xs)
    end
    @testset "$T randn" for T in SIGNED_TYPES
        xs = randn(rng, T, 1000)
        @test all(isfinite, xs)
        @test any(x -> x != zero(T), xs)
    end
end

@testset "eps / round / issubnormal" begin
    @test eps(Float8_E4M3(1.0)) == Float8_E4M3(0.125)
    @test eps(Float8_E4M3(2.0)) == Float8_E4M3(0.25)
    @test eps(Float8_E4M3(0.5)) == Float8_E4M3(0.0625)

    @test round(Float8_E4M3(1.5), RoundDown)    === Float8_E4M3(1.0)
    @test round(Float8_E4M3(1.5), RoundUp)      === Float8_E4M3(2.0)
    @test round(Float8_E4M3(2.5), RoundNearest) === Float8_E4M3(2.0)
    @test round(Float8_E4M3(0.5), RoundNearest) === Float8_E4M3(0.0)

    @test !issubnormal(zero(Float8_E4M3))
    @test !issubnormal(one(Float8_E4M3))
    @test  issubnormal(reinterpret(Float8_E4M3, 0x01))
    @test  issubnormal(reinterpret(Float8_E4M3, 0x07))
    @test !issubnormal(reinterpret(Float8_E4M3, 0x08))
    @test  issubnormal(-reinterpret(Float8_E4M3, 0x01))
end

@testset "exponent" begin
    # normals: matches Base.exponent on the round-tripped value
    @testset "$T normals" for T in (Float8_E3M4, Float8_E4M3, Float8_E5M2, Float8_E4M3FN)
        for u in 0x01:0xff
            x = reinterpret(T, u)
            (isnan(x) || isinf(x) || iszero(x) || issubnormal(x)) && continue
            @test exponent(x) == exponent(Float64(x))
        end
    end

    # subnormals: leading-1 position determines the unbiased exponent
    @test exponent(reinterpret(Float8_E4M3, 0x01)) == -9   # 0.001 × 2^-6
    @test exponent(reinterpret(Float8_E4M3, 0x02)) == -8   # 0.010 × 2^-6
    @test exponent(reinterpret(Float8_E4M3, 0x04)) == -7   # 0.100 × 2^-6
    @test exponent(reinterpret(Float8_E5M2, 0x01)) == -16  # 0.01  × 2^-14
    @test exponent(reinterpret(Float8_E5M2, 0x02)) == -15  # 0.10  × 2^-14

    # DomainError for zero / Inf / NaN — matches Base.exponent semantics
    @test_throws DomainError exponent(zero(Float8_E4M3))
    @test_throws DomainError exponent(-zero(Float8_E4M3))
    @test_throws DomainError exponent(inf(Float8_E5M2))
    @test_throws DomainError exponent(-inf(Float8_E5M2))
    @test_throws DomainError exponent(nan(Float8_E4M3))
end

@testset "sign_bits / exponent_bits / significand_bits / bitwidth (Base floats)" begin
    using Microfloats: sign_bits, exponent_bits, significand_bits, bitwidth

    @test (sign_bits(Float64), exponent_bits(Float64), significand_bits(Float64)) == (1, 11, 52)
    @test (sign_bits(Float32), exponent_bits(Float32), significand_bits(Float32)) == (1,  8, 23)
    @test (sign_bits(Float16), exponent_bits(Float16), significand_bits(Float16)) == (1,  5, 10)
    @test (sign_bits(BFloat16), exponent_bits(BFloat16), significand_bits(BFloat16)) == (1, 8, 7)

    @test bitwidth(Float64)  == 64
    @test bitwidth(Float32)  == 32
    @test bitwidth(Float16)  == 16
    @test bitwidth(BFloat16) == 16
end

@testset "Cross-microfloat arithmetic is unsupported" begin
    a = Float8_E4M3FN(1.0)
    b = Float8_E5M2(1.0)
    # No cross-microfloat promote_rule → Julia's promotion machinery errors.
    @test_throws ErrorException a + b
    @test_throws ErrorException a * b
    # Same-type still works.
    @test a + a == Float8_E4M3FN(2.0)
end
