using Microfloats: cvt, cvt_generic

# Result-or-throw comparator: conversions must agree on values *and* on
# whether they throw.
_cvt_outcome(f) = try f() catch e; (e isa DomainError || e isa ArgumentError) ? :threw : rethrow() end

@testset "conversion funnel" begin
    @testset "cvt ≡ constructors" begin
        for T in TYPES, x in (0.0, -0.0, 0.4, 1.0, 1.5, -2.75, 100.0, 1e6, -1e6, Inf, -Inf, NaN),
            mode in (RoundNearest, RoundToZero, RoundUp, RoundDown, RoundFromZero, RoundNearestTiesAway),
            pol in (Microfloats.SAT, Microfloats.OVF)

            a = _cvt_outcome(() -> T(x, mode; overflow=pol))
            b = _cvt_outcome(() -> cvt(T, Float32(x), mode, pol))
            @test a === b
        end
    end

    @testset "table specializations ≡ generic path" begin
        # Built-in pairs have @cvt_table-generated methods; they must agree
        # with the runtime generic path for every bit pattern, mode, policy.
        bad = []
        for S in TYPES_BUILTIN, T in TYPES_BUILTIN,
            mode in (RoundNearest, RoundToZero, RoundUp, RoundDown, RoundFromZero, RoundNearestTiesAway),
            pol in (Microfloats.SAT, Microfloats.OVF)

            for raw in 0x00:UInt8(2^bitwidth(S) - 1)
                x = reinterpret(S, raw)
                a = _cvt_outcome(() -> cvt(T, x, mode, pol))
                b = _cvt_outcome(() -> cvt_generic(T, Float32(x), mode, pol))
                a === b || push!(bad, (S, T, raw, mode, pol, a, b))
            end
        end
        @test isempty(bad)
    end

    @testset "@cvt_table on user-defined types" begin
        # Types defined long after Microfloats loaded — exercises the
        # world-age design of the generated table methods.
        Microfloats.@cvt_table UFloat5_E2M3 => Float8_E4M3
        Microfloats.@cvt_table Float8_E4M3 => UFloat5_E2M3
        for raw in 0x00:UInt8(2^bitwidth(UFloat5_E2M3) - 1)
            x = reinterpret(UFloat5_E2M3, raw)
            @test cvt(Float8_E4M3, x, RoundNearest, Microfloats.SAT) ===
                  cvt_generic(Float8_E4M3, Float32(x), RoundNearest, Microfloats.SAT)
        end
        # signed → unsigned: table generation hits throwing entries and must
        # preserve the runtime error path
        for raw in 0x00:0xff
            x = reinterpret(Float8_E4M3, UInt8(raw))
            a = _cvt_outcome(() -> cvt(UFloat5_E2M3, x, RoundNearest, Microfloats.SAT))
            b = _cvt_outcome(() -> cvt_generic(UFloat5_E2M3, Float32(x), RoundNearest, Microfloats.SAT))
            @test a === b
        end
    end

    @testset "bit-twiddling specializations ≡ generic path" begin
        # scalar twiddle: exact widening, so every mode/policy must agree
        for T in (Float8_E4M3, Float8_E4M3FN),
            mode in (RoundNearest, RoundToZero, RoundUp, RoundDown),
            pol in (Microfloats.SAT, Microfloats.OVF)

            for raw in 0x00:0x0f
                x = reinterpret(Float4_E2M1FN, raw)
                @test cvt(T, x, mode, pol) === cvt_generic(T, Float32(x), mode, pol)
            end
        end
        # packed → packed twiddle vs lanewise reference, all storage patterns
        NV, NA = Microfloats.NVector, Microfloats.NArray
        for T in (Float8_E4M3, Float8_E4M3FN)
            bad = 0
            for raw in 0x00:0xff
                xs = NA{Float4_E2M1FN,1,Tuple{2},UInt8}(raw)
                a = cvt(NV{T,2}, xs, RoundNearest, Microfloats.SAT)
                b = Microfloats.cvt_lanes(NV{T,2}, Tuple(xs), RoundNearest, Microfloats.SAT)
                bad += a !== b
            end
            for raw in 0x0000:0xffff
                xs = NA{Float4_E2M1FN,1,Tuple{4},UInt16}(UInt16(raw))
                a = cvt(NV{T,4}, xs, RoundNearest, Microfloats.SAT)
                b = Microfloats.cvt_lanes(NV{T,4}, Tuple(xs), RoundNearest, Microfloats.SAT)
                bad += a !== b
            end
            @test bad == 0
        end
        # entry constructor reaches the packed path
        f4 = Microfloats.Float4x2_E2M1FN((Float4_E2M1FN(0.5), Float4_E2M1FN(-6)))
        @test Tuple(Microfloats.Float8x2_E4M3FN(f4)) == (Float8_E4M3FN(0.5), Float8_E4M3FN(-6))
    end

    @testset "error hooks" begin
        @test_throws DomainError Float8_E8M0FNU(-1.0)
        @test_throws DomainError Float8_E8M0FNU(-0.0)
        @test_throws DomainError Float4_E2M1FN(NaN)
        @test_throws DomainError Float4_E2M1FN(Inf; overflow=Microfloats.OVF)
        @test_throws ArgumentError Float8_E4M3(1.0, RoundNearestTiesUp)
    end

    @testset "vector funnel" begin
        SV, NV = Microfloats.SVector, Microfloats.NVector

        # entry constructors resolve defaults and route through cvt
        v = Microfloats.Float8x2_E4M3FN(SV{2,Float32}(1.0f0, 0.99f0))
        @test Tuple(v) == (Float8_E4M3FN(1), Float8_E4M3FN(1))

        # rounding mode and overflow policy per call
        v = Microfloats.Float8x2_E4M3FN(SV{2,Float32}(0.99f0, 1f9), RoundToZero; overflow=Microfloats.SAT)
        @test Tuple(v) == (Float8_E4M3FN(0.9375), floatmax(Float8_E4M3FN))

        # sources: Float64/Float16 SVectors, packed NVectors, same-type repack
        @test Tuple(Microfloats.Float4x2_E2M1FN(SV{2,Float64}(1.0, 2.0))) ==
              (Float4_E2M1FN(1), Float4_E2M1FN(2))
        f4 = Microfloats.Float4x2_E2M1FN(SV{2,Float16}(1, 2))
        @test Tuple(Microfloats.Float6x2_E2M3FN(f4)) == (Float6_E2M3FN(1), Float6_E2M3FN(2))
        @test Microfloats.Float4x2_E2M1FN(f4) === f4

        # lanewise default agrees with scalar funnel, including policy
        xs = (10000f0, -1.5f0, 0.1f0, NaN32)
        a = Microfloats.cvt_lanes(NV{Float8_E5M2,4}, xs, RoundNearest, Microfloats.OVF)
        @test Tuple(a) === map(x -> Float8_E5M2(x; overflow=Microfloats.OVF), xs)

        # x4 entry constructor
        v4 = Microfloats.Float6x4_E2M3FN(SV{4,Float32}(0f0, 1f0, 1.5f0, 2f0))
        @test Tuple(v4) == (Float6_E2M3FN(0), Float6_E2M3FN(1), Float6_E2M3FN(1.5), Float6_E2M3FN(2))
    end
end
