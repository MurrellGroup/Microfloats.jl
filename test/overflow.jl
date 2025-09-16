@testset "Overflow" begin

    @testset "Has Inf+NaN" begin
        @testset for T in (
            Float8_E5M2, Float8_E4M3, Float8_E3M4, Float6_E3M2, Float6_E2M3, Float4_E2M1,
            MX_E5M2,
        )
            @test T(NaN, SAT) |> isnan
            @test T(NaN, OVF) |> isnan

            @test T(+Inf, SAT) == +floatmax(T)
            @test T(-Inf, SAT) == -floatmax(T)
            @test T(+Inf, OVF) == +Inf
            @test T(-Inf, OVF) == -Inf

            greater_than_floatmax = nextfloat(BFloat16(floatmax(T)))
            @test T(+greater_than_floatmax, SAT) == +floatmax(T)
            @test T(-greater_than_floatmax, SAT) == -floatmax(T)
            @test T(+greater_than_floatmax, OVF) == +Inf
            @test T(-greater_than_floatmax, OVF) == -Inf
        end
    end

    @testset "Has NaN" begin
        @testset for T in (
            MX_E4M3, MX_E8M0,
        )
            @test T(NaN, SAT) |> isnan
            @test T(NaN, OVF) |> isnan

            @test T(+Inf, SAT) == +floatmax(T)
            @test T(-Inf, SAT) == -floatmax(T)
            @test T(+Inf, OVF) |> isnan
            @test T(-Inf, OVF) |> isnan

            greater_than_floatmax = nextfloat(BFloat16(floatmax(T)))
            @test T(+greater_than_floatmax, SAT) == +floatmax(T)
            @test T(-greater_than_floatmax, SAT) == -floatmax(T)
            @test T(+greater_than_floatmax, OVF) |> isnan
            @test T(-greater_than_floatmax, OVF) |> isnan
        end
    end

    @testset "Finite" begin
        @testset for T in (
            MX_E3M2, MX_E2M3, MX_E2M1,
        )

            @test_throws DomainError T(NaN, SAT)
            @test_throws DomainError T(NaN, OVF)

            @test T(+Inf, SAT) == +floatmax(T)
            @test T(-Inf, SAT) == -floatmax(T)
            @test_throws DomainError T(+Inf, OVF)
            @test_throws DomainError T(-Inf, OVF)

            greater_than_floatmax = nextfloat(BFloat16(floatmax(T)))
            @test T(+greater_than_floatmax, SAT) == +floatmax(T)
            @test T(-greater_than_floatmax, SAT) == -floatmax(T)
            @test_throws DomainError T(+greater_than_floatmax, OVF)
            @test_throws DomainError T(-greater_than_floatmax, OVF)
        end

        @test MX_E2M1(6, SAT) == 6
        @test MX_E2M1(6, OVF) == 6
        @test MX_E2M1(7, SAT) == 6
        @test_throws DomainError MX_E2M1(7, OVF)
    end

end