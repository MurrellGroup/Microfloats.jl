module cuDNNExt

import Microfloats
import cuDNN

for (name, cudnn_name) in (
    :Float8_E4M3FN  => :CUDNN_DATA_FP8_E4M3,
    :Float8_E5M2    => :CUDNN_DATA_FP8_E5M2,
    :Float8_E8M0FNU => :CUDNN_DATA_FP8_E8M0,
    :Float4_E2M1FN  => :CUDNN_DATA_FP4_E2M1,
)
    if isdefined(cuDNN, cudnn_name)
        @eval cuDNN.cudnnDataType(::Type{Microfloats.$name}) = cuDNN.$cudnn_name
    end
end

end
