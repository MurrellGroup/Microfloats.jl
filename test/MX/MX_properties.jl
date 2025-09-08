@testset "MX: no Infs" begin
    for T in (MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1, MX_E8M0)
        @testset "$T no isinf()" begin
            for u in UInt8(0):UInt8(0xff)
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
        mm = UInt8(Microfloats.mantissa_mask(T))
        sm = UInt8(Microfloats.sign_mask(T))
        mo = Microfloats.mantissa_offset(T)
        nm = Microfloats.n_mantissa_bits(T)
        maxm = UInt8((UInt16(1) << nm) - 1)
        for s in (UInt8(0), sm)
            for mv in UInt8(0):maxm
                m = (mv << mo) & mm
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
            mo = Microfloats.mantissa_offset(T)
            nm = Microfloats.n_mantissa_bits(T)
            mm = UInt8(Microfloats.mantissa_mask(T))
            maxm = UInt8((UInt16(1) << nm) - 1)
            for s in (UInt8(0), sm)
                for mv in UInt8(0):maxm
                    m = (mv << mo) & mm
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
            used_mask = UInt8(Microfloats.sign_mask(T) | Microfloats.exponent_mask(T) | Microfloats.mantissa_mask(T))
            for u in UInt8(0):UInt8(0xff)
                # Only test canonical encodings where mantissa padding bits are zero
                (u & ~used_mask) != 0x00 && continue
                x = reinterpret(T, u)
                y = T(Float32(x))
                @test y ≡ x
            end
        end
    end
end

@testset "MX: saturation and NaN/Inf mapping from Float32" begin
    for T in (MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1, MX_E8M0)
        @testset "$T" begin
            fmax = floatmax(T)
            # +Inf/-Inf map to ±floatmax (unsigned maps both to +floatmax)
            if !(T <: MX_E5M2)
                @test T(Inf32) == fmax
                if Microfloats.n_sign_bits(T) == 0
                    @test T(-Inf32) == fmax
                else
                    @test T(-Inf32) == -fmax
                end
            end
            # NaN maps to sentinel for E4M3/E8M0, else saturates to floatmax
            if T <: Union{MX_E4M3, MX_E5M2, MX_E8M0}
                @test isnan(T(NaN32))
            else
                @test iszero(T(NaN32))
            end
            # Values just beyond floatmax saturate
            big = nextfloat(Float32(fmax))
            @test T(big) == fmax
            if Microfloats.n_sign_bits(T) == 0
                @test T(-big) == fmax
            else
                @test T(-big) == -fmax
            end
        end
    end
end

@testset "MX: subnormals and zeros" begin
    for T in (MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1)
        @testset "$T subnormal min value" begin
            if Microfloats.has_mantissa(T)
                u = UInt8(1) << Microfloats.mantissa_offset(T)
                x = reinterpret(T, u)
                expected = Float32(2.0)^(1 - Microfloats.exponent_bias(T) - Microfloats.n_mantissa_bits(T))
                @test Float32(x) == expected
            end
        end
    end
    for T in (MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1)
        @testset "$T signed zeros" begin
            if Microfloats.n_sign_bits(T) == 1
                zp = reinterpret(T, 0x00)
                zn = reinterpret(T, UInt8(Microfloats.sign_mask(T)))
                @test iszero(zp)
                @test iszero(zn)
                @test zp == zn
                @test signbit(zn) != signbit(zp)
            end
        end
    end
    @testset "E8M0 zero" begin
        @test iszero(reinterpret(MX_E8M0, 0x00))
    end
end

@testset "MX: equality and Bool semantics" begin
    for T in (MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1)
        @testset "$T" begin
            x = reinterpret(T, UInt8(1) << Microfloats.mantissa_offset(T))
            @test x != -x
            @test Bool(zero(T)) == false
            @test Bool(one(T)) == true
            @test_throws InexactError Bool(floatmax(T))
        end
    end
    @testset "E4M3 NaN equality" begin
        T = MX_E4M3
        x = reinterpret(T, UInt8(Microfloats.exponent_mask(T) | Microfloats.mantissa_mask(T)))
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
            for u in UInt8(0):UInt8(0xff)
                x = reinterpret(T, u)
                isnan(x) && continue
                # Only include canonical encodings: padding bits outside fields are zero
                used_mask = UInt8(Microfloats.sign_mask(T) | Microfloats.exponent_mask(T) | Microfloats.mantissa_mask(T))
                (u & ~used_mask) != 0x00 && continue
                push!(vals, (u, Float32(x), x))
            end
            sort!(vals, by = t -> t[2])
            for i in 1:length(vals)-1
                a = vals[i]; b = vals[i+1]
                if a[2] == b[2]
                    @test iszero(a[3]) && iszero(b[3]) && Microfloats.n_sign_bits(T) == 1
                else
                    @test a[2] < b[2]
                end
            end
        end
    end
end