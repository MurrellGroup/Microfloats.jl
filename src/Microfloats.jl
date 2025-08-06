module Microfloats

import Base: (-),(==),(<),(<=),isless,bitstring,
            isnan,iszero,one,zero,abs,
            floatmin,floatmax,typemin,typemax,
            Float32,Bool,
            (+), (-), (*), (/), (\), (^),
            sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,asinh,acosh,
            atanh,exp,exp2,exp10,expm1,log,log2,log10,sqrt,cbrt,log1p,
            atan,hypot,round,show,nextfloat,prevfloat,eps,
            promote_rule, sign, signbit

include("Microfloat.jl")
export Microfloat
export SignlessMicrofloat
export FiniteMicrofloat

include("microfloat_to_float32.jl")
include("float32_to_microfloat.jl")

end
