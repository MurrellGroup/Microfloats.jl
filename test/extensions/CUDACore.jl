using CUDACore:
    cudaDataType,
    R_8F_E4M3,
    R_8F_E5M2,
    R_8F_UE8M0,
    R_6F_E2M3,
    R_6F_E3M2,
    R_4F_E2M1

using Microfloats

@testset "CUDACore extension" begin
    @testset "cudaDataType" begin
        @test convert(cudaDataType, Float8_E4M3FN)  == R_8F_E4M3
        @test convert(cudaDataType, Float8_E5M2)    == R_8F_E5M2
        @test convert(cudaDataType, Float8_E8M0FNU) == R_8F_UE8M0
        @test convert(cudaDataType, Float6_E2M3FN)  == R_6F_E2M3
        @test convert(cudaDataType, Float6_E3M2FN)  == R_6F_E3M2
        @test convert(cudaDataType, Float4_E2M1FN)  == R_4F_E2M1
    end
end