module CUDACoreExt

using CUDACore:
    cudaDataType,
    R_8F_E4M3,
    R_8F_E5M2,
    R_8F_UE8M0,
    R_6F_E2M3,
    R_6F_E3M2,
    R_4F_E2M1
    # R_8F_UE5M3

using Microfloats:
    Float8_E4M3FN,
    Float8_E5M2,
    Float8_E8M0FNU,
    Float6_E2M3FN,
    Float6_E3M2FN,
    Float4_E2M1FN,
    Float8_E5M3FNU

Base.convert(::Type{cudaDataType}, ::Type{Float8_E4M3FN})  = R_8F_E4M3
Base.convert(::Type{cudaDataType}, ::Type{Float8_E5M2})    = R_8F_E5M2
Base.convert(::Type{cudaDataType}, ::Type{Float8_E8M0FNU}) = R_8F_UE8M0
Base.convert(::Type{cudaDataType}, ::Type{Float6_E2M3FN})  = R_6F_E2M3
Base.convert(::Type{cudaDataType}, ::Type{Float6_E3M2FN})  = R_6F_E3M2
Base.convert(::Type{cudaDataType}, ::Type{Float4_E2M1FN})  = R_4F_E2M1
# Base.convert(::Type{cudaDataType}, ::Type{Float8_E5M3FNU}) = R_8F_UE5M3

end
