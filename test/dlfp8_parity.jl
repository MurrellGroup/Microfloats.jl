import DLFP8Types

@testset "DLFP8Types.jl parity" begin
    # DLFP8Types.Float8_E4M3FN and .Float8_E5M2 use the same bit layout and
    # non-finite semantics as ours. Every bit pattern must agree on the
    # Float32 value (NaN encodings and signed zeros are allowed to differ
    # between the two packages — the semantic float value is what matters).
    @testset "$(M_T)" for (M_T, D_T) in (
        (Microfloats.Float8_E4M3FN, DLFP8Types.Float8_E4M3FN),
        (Microfloats.Float8_E5M2,   DLFP8Types.Float8_E5M2),
    )
        for i in 0x00:0xff
            mx = reinterpret(M_T, i)
            dx = reinterpret(D_T, i)
            @test Float32(mx) ≡ Float32(dx)
            @test isinf(mx) == isinf(dx)
            @test isnan(mx) == isnan(dx)
            @test iszero(mx) == iszero(dx)
        end
        # Float32 → narrow goes to the same semantic value.
        # DLFP8Types uses strict spec semantics (overflow → NaN for E4M3FN,
        # overflow → Inf for E5M2), which is our `OVF` policy.
        for f in Float32[0.0, -0.0, 1.0, -1.0, 3.5, -3.5, 448.0, 1000.0,
                         1.5f-5, 2.0f-7, NaN32, Inf32, -Inf32]
            @test Float32(M_T(f, OVF)) ≡ Float32(D_T(f))
        end
    end
end
