using Microfloats: raw_bits

# Bits above a type's width, which `reinterpret` can set, must not change any
# result, nor reach the bits of one.
@testset "Bits above the width" begin
    same(a, b) = raw_bits(a) === raw_bits(b)

    for T in TYPES
        N = bitwidth(T)
        N == 8 && continue
        for raw in 0x00:UInt8(2^N - 1), high in (0x01, 0xff)
            x = reinterpret(T, raw)
            y = reinterpret(T, raw | (high << N))

            @test raw_bits(y) === raw
            @test reinterpret(Unsigned, y) === raw
            @test repr(y) == repr(x)

            @test isnan(y) == isnan(x)
            @test isinf(y) == isinf(x)
            @test iszero(y) == iszero(x)
            @test signbit(y) == signbit(x)
            @test issubnormal(y) == issubnormal(x)
            @test y ≡ x
            @test hash(y) == hash(x)

            @test same(abs(y), abs(x))
            sign_bits(T) == 1 && @test same(-y, -x)
            @test same(nextfloat(y), nextfloat(x))
            @test same(prevfloat(y), prevfloat(x))

            for F in (Float16, BFloat16, Float32, Float64)
                @test F(y) === F(x)
            end
            @test Microfloats.cvt_generic(Float32, y) === Microfloats.cvt_generic(Float32, x)
            @test same(Float8_E4M3FN(y), Float8_E4M3FN(x))
            @test same(T(Float32(y)), T(Float32(x)))
        end
    end
end

# Nor do operations ever set them.
@testset "Results fit the width" begin
    fits(y) = reinterpret(UInt8, y) >> bitwidth(typeof(y)) == 0
    sources = Float32[0, -0.0, 1e-30, -1e-30, 0.3, -0.3, 1, -1, 1e30, -1e30, Inf, -Inf, NaN]

    for T in TYPES
        N = bitwidth(T)
        N == 8 && continue
        for f in (floatmax, typemax, typemin, zero, one)
            @test fits(f(T))
        end
        for raw in 0x00:UInt8(2^N - 1)
            x = reinterpret(T, raw)
            @test fits(abs(x))
            sign_bits(T) == 1 && @test fits(-x)
            @test fits(nextfloat(x))
            @test fits(prevfloat(x))
            @test fits(T(Float32(x)))
        end
        for f in sources, mode in (RoundNearest, RoundToZero, RoundUp, RoundDown)
            y = try
                T(f, mode)
            catch
                continue
            end
            @test fits(y)
        end
    end

    # Without Inf, the ends are floatmax and -floatmax, as Inf and -Inf are in Base.
    @testset "nextfloat and prevfloat stop at the ends" for T in TYPES
        @test nextfloat(typemax(T)) === typemax(T)
        @test prevfloat(typemin(T)) === typemin(T)
    end
end
