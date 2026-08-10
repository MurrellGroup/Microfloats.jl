module cuDNNExt

using cuDNN:
    CUDNN_DATA_FP8_E4M3,
    CUDNN_DATA_FP8_E5M2,
    CUDNN_DATA_FP8_E8M0,
    CUDNN_DATA_FP4_E2M1

using Microfloats:
    Float8_E4M3FN,
    Float8_E5M2,
    Float8_E8M0FNU,
    Float4_E2M1FN

cuDNN.cudnnDataType(::Type{Float8_E4M3FN}) = CUDNN_DATA_FP8_E4M3
cuDNN.cudnnDataType(::Type{Float8_E5M2}) = CUDNN_DATA_FP8_E5M2
cuDNN.cudnnDataType(::Type{Float8_E8M0FNU}) = CUDNN_DATA_FP8_E8M0
cuDNN.cudnnDataType(::Type{Float4_E2M1FN}) = CUDNN_DATA_FP4_E2M1

end
