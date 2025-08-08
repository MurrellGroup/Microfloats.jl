using Test
using Microfloats
using Random

const TYPES = [
    Microfloat(0, 3, 4),
    Microfloat(0, 4, 3),
    Microfloat(0, 3, 3),
    Microfloat(0, 4, 2),
    Microfloat(0, 5, 1),
    Microfloat(0, 3, 2),
    Microfloat(0, 2, 3),
    Microfloat(0, 2, 2),
    Microfloat(0, 3, 1),
    Microfloat(0, 1, 3),
    Microfloat(0, 2, 1),
    Microfloat(1, 3, 4),
    Microfloat(1, 4, 3),
    Microfloat(1, 3, 3),
    Microfloat(1, 4, 2),
    Microfloat(1, 5, 1),
    Microfloat(1, 3, 2),
    Microfloat(1, 2, 3),
    Microfloat(1, 2, 2),
    Microfloat(1, 3, 1),
    Microfloat(1, 1, 3),
    Microfloat(1, 2, 1),
]

@testset "Microfloat" begin
    @test UnsignedMicrofloat(3, 4) == Microfloat(0, 3, 4)


    @testset for T in TYPES
        @test hash(one(T)) == hash(1)

        @test prevfloat(eps(T)) < eps(T)
        @test nextfloat(eps(T)) > eps(T)
        @test nextfloat(zero(T)) > zero(T)
        @test isfinite(prevfloat(T(Inf)))

        if Microfloats.n_exponent_bits(T) > 1
            @test floatmin(T) == reinterpret(T, 0x01 << Microfloats.exponent_offset(T))
        else
            @test_throws DomainError floatmin(T)
        end
        @test floatmax(T) == prevfloat(T(Inf))

        @test typemax(T) == T(Inf)

        @test sign(T(Inf)) == 1.0
        @test sign(T(1.0)) == 1.0
        @test sign(T(0.0)) == 0.0
        @test isnan(sign(T(NaN)))

        if T <: SignedMicrofloat
            @test typemin(T) == T(-Inf)

            @test sign(T(-0.0)) == -0.0
            @test sign(T(-1.0)) == -1.0
            @test sign(T(-Inf)) == -1.0
        else
            @test typemin(T) == zero(T)
        end

        @test precision(T) == Microfloats.n_mantissa_bits(T) + 1
    end
end

@testset "IEEE microfloats: subnormals and rounding" begin
    @testset for T in TYPES
        bias = Microfloats.bias(T)
        M = Microfloats.n_mantissa_bits(T)
        mo = Microfloats.mantissa_offset(T)
        # Encoding for the minimum positive subnormal (mantissa LSB only)
        min_sub_u = UInt8(1) << mo
        min_sub = reinterpret(T, min_sub_u)

        # Real values
        min_sub_val = Float32(2.0)^(1 - bias - M)
        half = min_sub_val/2
        just_below_half = prevfloat(half)
        just_above_half = nextfloat(half)
        just_below = prevfloat(min_sub_val)
        just_above = nextfloat(min_sub_val)

        # Exact min subnormal
        @test Float32(min_sub) == min_sub_val

        # Values well below half of min subnormal should round to +0
        @test T(half/4) == zero(T)

        # Exactly half rounds to even -> zero; below half also zero
        @test T(half) == zero(T)
        @test T(just_below_half) == zero(T)

        # Values just above half of min subnormal should round to min subnormal
        @test T(just_above_half) == min_sub

        # Values just below min subnormal remain min subnormal after rounding up from Float32
        @test T(just_below) == min_sub

        # Values just above min subnormal quantize to min subnormal or the next representable
        # depending on spacing; at least should be >= min_sub
        @test Float32(T(just_above)) >= min_sub_val
    end
end

@testset "IEEE microfloats: monotonic Float32 mapping (canonical encodings)" begin
    @testset for T in TYPES
        vals = Tuple{UInt8,Float32,Any}[]
        mshift = Microfloats.mantissa_offset(T)
        mmask  = UInt8(Microfloats.mantissa_mask(T))
        for u in UInt8(0):UInt8(0xff)
            x = reinterpret(T, u)
            isnan(x) && continue
            # Only include canonical encodings: mantissa padding bits zero
            ((u & ~mmask) != (u & ~mmask & ~(UInt8(1)<<mshift - UInt8(1)))) && continue
            push!(vals, (u, Float32(x), x))
        end
        sort!(vals, by = t -> t[2])
        for i in 1:length(vals)-1
            a = vals[i]; b = vals[i+1]
            if a[2] == b[2]
                # duplicate comes only from signed zeros
                @test iszero(a[3]) && iszero(b[3])
            else
                @test a[2] < b[2]
            end
        end
    end
end

@testset "IEEE microfloats: rand and randn" begin
    rng = MersenneTwister(123)
    @testset for T in TYPES
        @testset "$T rand()" begin
            xs = rand(rng, T, 1000)
            @test all(x -> isfinite(x), xs)
            @test any(x -> x != zero(T), xs)  # likely non-degenerate
        end
        if Microfloats.n_sign_bits(T) == 1
            @testset "$T randn()" begin
                xs = randn(rng, T, 1000)
                @test all(isfinite, xs)
                @test any(x -> x != zero(T), xs)
            end
        end
    end
end
