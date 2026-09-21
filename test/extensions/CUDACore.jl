using CUDACore: CUDACore, cudaDataType

using Microfloats

@testset "CUDACore extension" begin
    @testset "cudaDataType" begin
        # CUDACore gained the narrow data types over several releases; the
        # extension defines a conversion for each constant that exists.
        for (T, name) in (
            Float8_E4M3FN  => :R_8F_E4M3,
            Float8_E5M2    => :R_8F_E5M2,
            Float8_E8M0FNU => :R_8F_UE8M0,
            Float6_E2M3FN  => :R_6F_E2M3,
            Float6_E3M2FN  => :R_6F_E3M2,
            Float4_E2M1FN  => :R_4F_E2M1,
            Float8_E5M3FNU => :R_8F_UE5M3,
        )
            isdefined(CUDACore, name) || continue
            @test convert(cudaDataType, T) == getfield(CUDACore, name)
        end
    end
end
