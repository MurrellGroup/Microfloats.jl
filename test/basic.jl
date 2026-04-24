@testset "Microfloat generic properties" begin
    @testset "$T" for T in TYPES
        @test hash(one(T)) == hash(1)
        @test precision(T) == Microfloats.significand_bits(T) + 1
        @test floatmax(T) > zero(T)
        @test isfinite(floatmax(T))

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

@testset "Cross-microfloat arithmetic is unsupported" begin
    a = Float8_E4M3FN(1.0)
    b = Float8_E5M2(1.0)
    # No cross-microfloat promote_rule → Julia's promotion machinery errors.
    @test_throws ErrorException a + b
    @test_throws ErrorException a * b
    # Same-type still works.
    @test a + a == Float8_E4M3FN(2.0)
end
