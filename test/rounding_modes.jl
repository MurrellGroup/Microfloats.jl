@testset "Default mode = RoundNearest" begin
    @testset "$T" for T in TYPES
        # Sample a few non-tie values; all three constructors must agree.
        for v in (0.0f0, 1.0f0, 2.0f0, 0.5f0)
            sign_bits(T) == 0 && v < 0 && continue
            @test T(v) === T(v, RoundNearest)
        end
    end
end

@testset "Round-half-to-even ties (Float8_E4M3)" begin
    # E4M3 spacing in [1,2) is 1/8. Midpoint(1.0, 1.125) = 1.0625.
    # Mantissa LSB: 1.0 even, 1.125 odd → ties-to-even → 1.0.
    @test Float8_E4M3(1.0625f0, RoundNearest)         === Float8_E4M3(1.0)
    @test Float8_E4M3(1.0625f0, RoundNearestTiesAway) === Float8_E4M3(1.125)
    @test Float8_E4M3(1.0625f0, RoundToZero)          === Float8_E4M3(1.0)

    # Midpoint(1.125, 1.25) = 1.1875.
    # Mantissa LSB: 1.125 odd, 1.25 even → ties-to-even → 1.25.
    @test Float8_E4M3(1.1875f0, RoundNearest)         === Float8_E4M3(1.25)
    @test Float8_E4M3(1.1875f0, RoundNearestTiesAway) === Float8_E4M3(1.25)
    @test Float8_E4M3(1.1875f0, RoundToZero)          === Float8_E4M3(1.125)
end

@testset "RoundUp / RoundDown / RoundFromZero (Float8_E4M3)" begin
    # 1.1 is in (1.0, 1.125), not exactly representable.
    @test Float8_E4M3( 1.1f0, RoundUp)       === Float8_E4M3( 1.125)
    @test Float8_E4M3( 1.1f0, RoundDown)     === Float8_E4M3( 1.0)
    @test Float8_E4M3( 1.1f0, RoundFromZero) === Float8_E4M3( 1.125)
    # RoundUp on a negative goes toward zero (less negative).
    @test Float8_E4M3(-1.1f0, RoundUp)       === Float8_E4M3(-1.0)
    @test Float8_E4M3(-1.1f0, RoundDown)     === Float8_E4M3(-1.125)
    @test Float8_E4M3(-1.1f0, RoundFromZero) === Float8_E4M3(-1.125)
    # Exactly representable: every mode is a no-op.
    for mode in (RoundUp, RoundDown, RoundFromZero)
        @test Float8_E4M3(1.5f0, mode) === Float8_E4M3(1.5)
        @test Float8_E4M3(-1.5f0, mode) === Float8_E4M3(-1.5)
    end
end

@testset "Unsupported rounding mode errors cleanly" begin
    # Non-IEEE-754-directed modes (e.g. NearestTiesUp) — we don't support them.
    # Must error, not recurse.
    @test_throws ArgumentError Float8_E4M3(1.5f0, RoundNearestTiesUp)
    @test_throws ArgumentError Float8_E4M3(1.5,   RoundNearestTiesUp)  # Float64 too
end

@testset "Truncation does not round up (Float4_E2M1FN)" begin
    # Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0.
    @test Float4_E2M1FN(2.5f0,  RoundToZero) === Float4_E2M1FN(2.0)
    @test Float4_E2M1FN(2.99f0, RoundToZero) === Float4_E2M1FN(2.0)
    @test Float4_E2M1FN(0.49f0, RoundToZero) === Float4_E2M1FN(0.0)
end

@testset "Ties-away rounds magnitude up (Float4_E2M1FN)" begin
    # 2.5 is the midpoint of 2.0 and 3.0. Both ties → away.
    @test Float4_E2M1FN(2.5f0, RoundNearestTiesAway) === Float4_E2M1FN(3.0)
    @test Float4_E2M1FN(1.25f0, RoundNearestTiesAway) === Float4_E2M1FN(1.5)
end

@testset "NaN propagation across modes" begin
    @testset "$T" for T in (Float8_E5M2, Float8_E4M3, Float8_E4M3FN)
        for mode in (RoundNearest, RoundNearestTiesAway, RoundToZero)
            @test isnan(T(NaN32, mode))
        end
    end
end

@testset "IEEE-754 mode-driven overflow" begin
    # Overflow target depends on the rounding mode, per IEEE-754. Modes that
    # round toward zero or toward floatmax-on-the-right-side saturate
    # regardless of type policy; modes that round toward Inf go to Inf when
    # the format has one.
    big = nextfloat(Float32(floatmax(Float8_E4M3)))  # E4M3 = IEEE + OVF
    @test Float8_E4M3( big, RoundNearest)         === +inf(Float8_E4M3)        # mode wants +Inf
    @test Float8_E4M3(-big, RoundNearest)         === -inf(Float8_E4M3)        # mode wants -Inf
    @test Float8_E4M3( big, RoundNearestTiesAway) === +inf(Float8_E4M3)
    @test Float8_E4M3( big, RoundFromZero)        === +inf(Float8_E4M3)
    @test Float8_E4M3( big, RoundToZero)          === +floatmax(Float8_E4M3)   # mode wants +floatmax
    @test Float8_E4M3(-big, RoundToZero)          === -floatmax(Float8_E4M3)
    @test Float8_E4M3( big, RoundUp)              === +inf(Float8_E4M3)        # positive: toward +Inf
    @test Float8_E4M3(-big, RoundUp)              === -floatmax(Float8_E4M3)   # negative: toward zero
    @test Float8_E4M3( big, RoundDown)            === +floatmax(Float8_E4M3)   # positive: toward zero
    @test Float8_E4M3(-big, RoundDown)            === -inf(Float8_E4M3)        # negative: toward -Inf
end

@testset "SAT policy clamps regardless of mode" begin
    big = nextfloat(Float32(floatmax(_E4M3FN_SAT)))
    @test _E4M3FN_SAT(big, RoundNearest)         === floatmax(_E4M3FN_SAT)
    @test _E4M3FN_SAT(big, RoundNearestTiesAway) === floatmax(_E4M3FN_SAT)
    @test _E4M3FN_SAT(big, RoundToZero)          === floatmax(_E4M3FN_SAT)
    @test _E4M3FN_SAT(big, RoundFromZero)        === floatmax(_E4M3FN_SAT)
    @test _E4M3FN_SAT(big, RoundUp)              === floatmax(_E4M3FN_SAT)
end

@testset "Overflow policy keyword override" begin
    big = nextfloat(Float32(floatmax(Float8_E4M3FN)))
    # Default policy for E4M3FN is OVF → NaN under RoundNearest.
    @test isnan(Float8_E4M3FN(big, RoundNearest))
    # Override to SAT for one call: clamp instead.
    @test Float8_E4M3FN(big, RoundNearest; overflow=SAT) === floatmax(Float8_E4M3FN)
    # No-mode entry also honors the keyword.
    @test Float8_E4M3FN(big; overflow=SAT) === floatmax(Float8_E4M3FN)
end

@testset "Negative inputs (signed types)" begin
    # Tie at -1.0625: away-from-zero means more negative.
    @test Float8_E4M3(-1.0625f0, RoundNearest)         === Float8_E4M3(-1.0)
    @test Float8_E4M3(-1.0625f0, RoundNearestTiesAway) === Float8_E4M3(-1.125)
    @test Float8_E4M3(-1.0625f0, RoundToZero)          === Float8_E4M3(-1.0)
end

@testset "Unsigned types reject negative inputs in every mode" begin
    for mode in (RoundNearest, RoundNearestTiesAway, RoundToZero)
        @test_throws DomainError Float8_E8M0FNU(-1.0f0, mode)
    end
end

@testset "Double-rounding regression (Float32 source)" begin
    # Float32(1.31640625) is strictly above the E4M3 midpoint 1.3125, so direct
    # rounding lands on 1.375. Double-rounding through an intermediate format
    # with round-to-even at each stage gives 1.25.
    @test Float8_E4M3(Float32(1.31640625)) === Float8_E4M3(1.375)

    # nextfloat(1.0078125f0) is strictly above the E1M7 midpoint 1.0078125 of
    # {1.0, 1.015625}; direct rounding goes up.
    v = nextfloat(1.0078125f0)
    @test Float64(_E1M7_FN(v)) == 1.015625
end
