module CUDACoreExt

import Microfloats
import CUDACore: CUDACore, cudaDataType

for (name, cuda_name) in (
    :Float8_E4M3FN  => :R_8F_E4M3,
    :Float8_E5M2    => :R_8F_E5M2,
    :Float8_E8M0FNU => :R_8F_UE8M0,
    :Float6_E2M3FN  => :R_6F_E2M3,
    :Float6_E3M2FN  => :R_6F_E3M2,
    :Float4_E2M1FN  => :R_4F_E2M1,
    :Float8_E5M3FNU => :R_8F_UE5M3,
)
    if isdefined(CUDACore, cuda_name)
        @eval Base.convert(::Type{cudaDataType}, ::Type{Microfloats.$name}) = CUDACore.$cuda_name
    end
end

end
