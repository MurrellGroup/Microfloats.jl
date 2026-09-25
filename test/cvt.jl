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

    @testset "@cvt_table methods ≡ generic path" begin
        # Built-in pairs have @cvt_table-generated methods (twiddles or tables); they must agree
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

    @testset "bit-twiddles" begin
        MODES = (RoundNearest, RoundToZero, RoundUp, RoundDown, RoundFromZero, RoundNearestTiesAway)
        evaluate(pieces, i) = (p = pieces[findlast(p -> p[1] <= i, pieces)]; p[2] * UInt8(i) + p[3])

        # The fitted pieces reproduce every table exactly, for every pair,
        # mode and policy, whether or not the cost admits a twiddle (the
        # device takes the lookup where the host takes the twiddle).
        bad = []
        for S in TYPES, T in TYPES, mode in MODES, pol in (Microfloats.SAT, Microfloats.OVF)
            table = Microfloats.cvt_generic_table(T, S, mode, pol)
            table === nothing && continue
            pieces, mask, shift = Microfloats.twiddle_fit(T, S, table)
            for raw in 0:length(table) - 1
                t = evaluate(pieces, raw & mask)
                if shift !== nothing
                    s = UInt8(raw) & Microfloats.sign_mask(S)
                    t |= shift >= 0 ? s << shift : s >> -shift
                end
                t == table[raw + 1] || push!(bad, (S, T, mode, pol, raw))
            end
        end
        @test isempty(bad)

        # Exact widenings are independent of mode and policy, and cheap.
        exact = (
            (Float4_E2M1FN, Float6_E2M3FN) => 1, (Float4_E2M1FN, Float6_E3M2FN) => 3,
            (Float4_E2M1FN, Float8_E3M4) => 3, (Float4_E2M1FN, Float8_E4M3) => 4,
            (Float4_E2M1FN, Float8_E4M3FN) => 4, (Float4_E2M1FN, Float8_E5M2) => 4,
            (Float6_E2M3FN, Float8_E3M4) => 4, (Float6_E2M3FN, Float8_E4M3) => 5,
            (Float6_E2M3FN, Float8_E4M3FN) => 5, (Float6_E3M2FN, Float8_E4M3) => 5,
            (Float6_E3M2FN, Float8_E4M3FN) => 5, (Float6_E3M2FN, Float8_E5M2) => 5,
        )
        for ((S, T), cost) in exact, mode in MODES, pol in (Microfloats.SAT, Microfloats.OVF)
            table = Microfloats.cvt_generic_table(T, S, mode, pol)
            @test Microfloats.twiddle_cost(Microfloats.twiddle_fit(T, S, table)) == cost
        end

        # Packed → packed for any length: the lanewise path unpacks, twiddles
        # each lane and packs. Exhaustive over storage for 2 and 4 lanes.
        NV = Microfloats.NVector
        for (S, T) in ((Float4_E2M1FN, Float8_E4M3FN), (Float4_E2M1FN, Float8_E5M2),
                       (Float4_E2M1FN, Float6_E2M3FN), (Float6_E2M3FN, Float8_E4M3))
            bad = 0
            mask = UInt8(2^bitwidth(S) - 1)
            for (N, U, raws) in ((2, UInt16, 0x0000:0x0fff), (4, UInt32, rand(UInt32, 2^14)),
                                 (8, UInt64, rand(UInt64, 2^12)))
                for raw in raws
                    lanes = ntuple(i -> reinterpret(S, (raw >> (8(i - 1))) % UInt8 & mask), N)
                    xs = NV{S,N}(lanes)
                    a = cvt(NV{T,N}, xs, RoundNearest, Microfloats.SAT)
                    b = ntuple(i -> cvt_generic(T, Float32(lanes[i]), RoundNearest, Microfloats.SAT), N)
                    bad += Tuple(a) !== b
                end
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

        # Both vector forms agree with the scalar funnel lane by lane, and the
        # packed default (convert unpacked, then pack) agrees with packing
        # lanewise conversions.
        xs = (0.3f0, -7.9f0, 1f9, -0.0f0)
        for T in (Float6_E2M3FN, Float6_E3M2FN, Float8_E4M3FN, Float8_E5M2),
            mode in (RoundNearest, RoundToZero, RoundUp, RoundDown)
            want = map(x -> cvt(T, x, mode, Microfloats.SAT), xs)
            @test Tuple(cvt(SV{4,T}, xs, mode, Microfloats.SAT)) === want
            @test cvt(NV{T,4}, xs, mode, Microfloats.SAT) ===
                  Microfloats.cvt_lanes(NV{T,4}, xs, mode, Microfloats.SAT)
            @test Tuple(SV{4,T}(SV{4,Float32}(xs), mode; overflow=Microfloats.SAT)) === want
        end
    end

    @testset "widening funnel" begin
        using Microfloats: SVector, NVector, WideFloat
        # exact: every value survives Float16/BFloat16/Float32/Float64 and
        # back, except where Float16's range is too narrow
        for T in TYPES, raw in 0x00:UInt8(2^bitwidth(T) - 1)
            x = reinterpret(T, raw)
            for F in (BFloat16, Float32, Float64)
                y = cvt(F, x)
                @test y isa F
                @test F(x) === y                                   # constructors are sugar
                @test cvt_generic(F, x) === y
                @test isnan(x) ? isnan(y) : isinf(x) ? (isinf(y) && signbit(y) == signbit(x)) :
                      T(y; overflow=Microfloats.SAT) === x
            end
            @test cvt(Float16, x) === Float16(cvt(Float32, x))
            @test isnan(x) || Float64(cvt(BFloat16, x)) == cvt(Float64, x)
        end

        # vector forms agree with the scalar funnel for every source container
        for T in TYPES_BUILTIN, F in (Float16, BFloat16, Float32, Float64)
            xs = ntuple(i -> reinterpret(T, UInt8(i * 7 % 2^bitwidth(T))), 4)
            want = SVector{4,F}(map(x -> cvt(F, x), xs))
            same(a, b) = all(map((p, q) -> isequal(p, q), Tuple(a), Tuple(b)))
            @test same(cvt(SVector{4,F}, xs), want)
            @test same(cvt(SVector{4,F}, SVector{4,T}(xs)), want)
            @test same(cvt(SVector{4,F}, NVector{T,4}(xs)), want)
            @test same(SVector{4,F}(SVector{4,T}(xs)), want)
            @test same(SVector{4,F}(NVector{T,4}(xs)), want)
            @test SVector{4,F}(NVector{T,4}(xs)) isa SVector{4,F}
        end
    end

    @testset "no method ambiguities" begin
        @test isempty(Test.detect_ambiguities(Microfloats; recursive=true))
    end
end
