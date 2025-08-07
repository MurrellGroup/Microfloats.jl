import Base: (<), (<=), isless

for op in (:<, :<=, :isless)
    @eval ($op)(a::Microfloat, b::Real) = ($op)(Float32(a), Float32(b))
    @eval ($op)(a::Real, b::Microfloat) = ($op)(Float32(a), Float32(b))
    @eval ($op)(a::Microfloat, b::Microfloat) = ($op)(Float32(a), Float32(b))
end


import Base: (+), (-), (*), (/), (\), (^)

for op in (:+, :-, :*, :/, :\, :^)
    @eval ($op)(a::Microfloat, b::Number) = promote_type(typeof(a), typeof(b))(($op)(Float32(a), Float32(b)))
    @eval ($op)(a::Number, b::Microfloat) = promote_type(typeof(a), typeof(b))(($op)(Float32(a), Float32(b)))
    @eval ($op)(a::Microfloat, b::Microfloat) = promote_type(typeof(a), typeof(b))(($op)(Float32(a), Float32(b)))
end

(^)(a::T, b::Integer) where T<:Microfloat = T(Float32(a)^b)


import Base: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh,
             atanh, exp, exp2, exp10, expm1, log, log2, log10, sqrt, cbrt, log1p

for func in (:sin,:cos,:tan,:asin,:acos,:atan,:sinh,:cosh,:tanh,:asinh,:acosh,
    :atanh,:exp,:exp2,:exp10,:expm1,:log,:log2,:log10,:sqrt,:cbrt,:log1p)
    @eval $func(a::T) where T<:Microfloat = T($func(Float32(a)))
end


import Base: atan, hypot

for func in (:atan,:hypot)
    @eval $func(a::T, b::T) where T<:Microfloat = T($func(Float32(a),Float32(b)))
end


import Base: frexp, ldexp

for func in (:frexp,:ldexp)
    @eval $func(a::T, b::Int) where T<:Microfloat = T($func(Float32(a),b))
end


import Base: modf, mod2pi

for func in (:modf,:mod2pi)
    @eval $func(a::T) where T<:Microfloat = T($func(Float32(a)))
end
