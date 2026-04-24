@testset "MX: no Infs" begin
    for T in (MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1, MX_E8M0)
        @testset "$T no isinf()" begin
            for u in 0x00:0xff
                @test !isinf(reinterpret(T, u))
            end
        end
    end
end

@testset "MX: special encodings" begin
    # E4M3: only mantissa=111 at exp=1111 is NaN; others finite
    @testset "E4M3" begin
        T = MX_E4M3
        em = UInt8(Microfloats.exponent_mask(T))
        mm = UInt8(Microfloats.significand_mask(T))
        sm = UInt8(Microfloats.sign_mask(T))
        nm = Microfloats.significand_bits(T)
        maxm = UInt8((UInt16(1) << nm) - 1)
        for s in (0x00, sm)
            for mv in 0x00:maxm
                m = mv & mm
                x = reinterpret(T, (s & sm) | em | m)
                if m == mm
                    @test isnan(x)
                else
                    @test isfinite(x)
                end
            end
        end
    end

    # E3M2/E2M3/E2M1: exp=all-ones are finite; no NaN sentinel
    for T in (MX_E3M2, MX_E2M3, MX_E2M1)
        @testset "$T exp=all-ones finite" begin
            em = UInt8(Microfloats.exponent_mask(T))
            sm = UInt8(Microfloats.sign_mask(T))
            nm = Microfloats.significand_bits(T)
            mm = UInt8(Microfloats.significand_mask(T))
            maxm = UInt8((UInt16(1) << nm) - 1)
            for s in (0x00, sm)
                for mv in 0x00:maxm
                    m = mv & mm
                    x = reinterpret(T, (s & sm) | em | m)
                    @test isfinite(x)
                    @test !isnan(x)
                end
            end
        end
    end

    # E8M0: 0xff is NaN
    @testset "E8M0" begin
        T = MX_E8M0
        @test isnan(reinterpret(T, 0xff))
        @test !isnan(reinterpret(T, 0x00))
        @test !isfinite(reinterpret(T, 0xff))
    end
end

@testset "MX: round-trip via Float32 preserves bits (canonical encodings)" begin
    for T in (MX_E4M3, MX_E5M2, MX_E3M2, MX_E2M3, MX_E2M1, MX_E8M0)
        @testset "$T" begin
            used_mask = Microfloats.sign_mask(T) | Microfloats.exponent_mask(T) | Microfloats.significand_mask(T)
            for u in 0x00:0xff
                (u & ~used_mask) != 0x00 && continue
                x = reinterpret(T, u)
                y = T(Float32(x))
                @test y ≡ x
            end
        end
    end
end

@testset "MX: default overflow mapping from Float32" begin
    # Each MX type's default policy is baked in. We test the shipped
    # behavior per type; alternate semantics require a twin type.
    @testset "E5M2 (IEEE, OVF)" begin
        T = MX_E5M2
        @test T(+Inf32) == inf(T)
        @test T(-Inf32) == -inf(T)
        @test isnan(T(NaN32))
        big = nextfloat(BFloat16(floatmax(T)))
        @test T(+big) == inf(T)
        @test T(-big) == -inf(T)
    end

    @testset "E4M3 (NanOnlyAllOnes, OVF)" begin
        T = MX_E4M3
        @test isnan(T(+Inf32))
        @test isnan(T(-Inf32))
        @test isnan(T(NaN32))
        big = nextfloat(BFloat16(floatmax(T)))
        @test isnan(T(+big))
        @test isnan(T(-big))
    end

    @testset "E8M0 (NanOnlyAllOnes, OVF)" begin
        T = MX_E8M0
        @test isnan(T(+Inf32))
        @test_throws DomainError T(-Inf32)
        @test isnan(T(NaN32))
        big = nextfloat(BFloat16(floatmax(T)))
        @test isnan(T(big))
    end

    @testset "FiniteOnly $T" for T in (MX_E3M2, MX_E2M3, MX_E2M1)
        fmax = floatmax(T)
        @test T(+Inf32) == +fmax
        @test T(-Inf32) == -fmax
        @test_throws DomainError T(NaN32)
        big = nextfloat(BFloat16(fmax))
        @test T(+big) == +fmax
        @test T(-big) == -fmax
    end
end

@testset "MX: subnormals and zeros" begin
    for T in (MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1)
        @testset "$T subnormal min value" begin
            if Microfloats.significand_bits(T) > 0
                x = reinterpret(T, 0x01)
                expected = Float32(2.0)^(1 - Microfloats.exponent_bias(T) - Microfloats.significand_bits(T))
                @test Float32(x) == expected
            end
        end
    end
    for T in (MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1)
        @testset "$T signed zeros" begin
            if Microfloats.sign_bits(T) == 1
                zp = reinterpret(T, 0x00)
                zn = reinterpret(T, Microfloats.sign_mask(T))
                @test iszero(zp)
                @test iszero(zn)
                @test zp == zn
                @test signbit(zn) != signbit(zp)
            end
        end
    end
    @testset "E8M0 minimum (no zero)" begin
        @test !iszero(reinterpret(MX_E8M0, 0x00))
        @test Float32(reinterpret(MX_E8M0, 0x00)) == 2f0^-127
    end
end

@testset "MX: equality and Bool semantics" begin
    for T in (MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1)
        @testset "$T" begin
            x = reinterpret(T, 0x01)
            @test x != -x
            @test Bool(zero(T)) == false
            @test Bool(one(T)) == true
            @test_throws InexactError Bool(floatmax(T))
        end
    end
    @testset "E4M3 NaN equality" begin
        T = MX_E4M3
        x = reinterpret(T, Microfloats.exponent_mask(T) | Microfloats.significand_mask(T))
        @test isnan(x)
        @test !(x == x)
    end
    @testset "E8M0 NaN equality" begin
        T = MX_E8M0
        x = reinterpret(T, 0xff)
        @test isnan(x)
        @test !(x == x)
    end
end

@testset "MX: Float32 mapping monotonic (canonical; ignoring signed zero duplicates)" begin
    for T in (MX_E4M3, MX_E5M2, MX_E3M2, MX_E2M3, MX_E2M1, MX_E8M0)
        @testset "$T" begin
            vals = Tuple{UInt8,Float32,Any}[]
            for u in 0x00:0xff
                x = reinterpret(T, u)
                isnan(x) && continue
                # Only include canonical encodings: padding bits outside fields are zero
                used_mask = UInt8(Microfloats.sign_mask(T) | Microfloats.exponent_mask(T) | Microfloats.significand_mask(T))
                (u & ~used_mask) != 0x00 && continue
                push!(vals, (u, Float32(x), x))
            end
            sort!(vals, by = t -> t[2])
            for i in 1:length(vals)-1
                a = vals[i]; b = vals[i+1]
                if a[2] == b[2]
                    @test iszero(a[3]) && iszero(b[3]) && Microfloats.sign_bits(T) == 1
                else
                    @test a[2] < b[2]
                end
            end
        end
    end
end