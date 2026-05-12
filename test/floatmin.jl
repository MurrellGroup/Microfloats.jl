# An E=1 microfloat with IEEE non-finite has biased-exponent 1 reserved for
# Inf/NaN, so the smallest-normal bit pattern (1 << M) coincides with the Inf
# encoding. `floatmin` must throw a `DomainError` rather than return Inf.

@microfloat _E1M7_IEEE sign=0 exponent=1 significand=7
@microfloat _E1M7_FN   sign=0 exponent=1 significand=7 nonfinite=FiniteOnly
@microfloat _E1M7_NaN  sign=0 exponent=1 significand=7 nonfinite=NanOnlyAllOnes

@testset "floatmin on degenerate (no-normals) formats" begin
    @test_throws DomainError floatmin(_E1M7_IEEE)

    # E=1 without IEEE non-finite still has one normal exponent.
    @test Float64(floatmin(_E1M7_FN))  == 2.0
    @test Float64(floatmin(_E1M7_NaN)) == 2.0
end

@testset "floatmin on shipped types" begin
    @test Float64(floatmin(Float8_E5M2))    == 2.0^-14
    @test Float64(floatmin(Float8_E4M3))    == 2.0^-6
    @test Float64(floatmin(Float8_E3M4))    == 2.0^-2
    @test Float64(floatmin(Float8_E4M3FN))  == 2.0^-6
    @test Float64(floatmin(Float4_E2M1FN))  == 1.0
    @test Float64(floatmin(Float6_E2M3FN))  == 1.0
    @test Float64(floatmin(Float6_E3M2FN))  == 0.25
    @test Float64(floatmin(Float8_E8M0FNU)) == 2.0^-127  # M=0 → returns 0x00 pattern
end
