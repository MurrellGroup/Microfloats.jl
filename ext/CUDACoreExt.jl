module CUDACoreExt

using Microfloats
using Microfloats: OverflowPolicy, SAT, OVF, overflow_policy, BFloat16,
                   Float8_E5M2, Float8_E4M3, Float8_E3M4,
                   Float8_E4M3FN, Float8_E8M0FNU,
                   Float6_E2M3FN, Float6_E3M2FN, Float4_E2M1FN,
                   Float8x2_E4M3FN, Float8x4_E4M3FN,
                   Float8x2_E5M2, Float8x4_E5M2,
                   Float8x2_E8M0FNU, Float8x4_E8M0FNU,
                   Float6x2_E2M3FN, Float6x4_E2M3FN,
                   Float6x2_E3M2FN, Float6x4_E3M2FN,
                   Float4x2_E2M1FN, Float4x4_E2M1FN
using BitPacking: NArray, NVector
using CUDACore
using CUDACore: @device_override, compute_capability, target_feature_set
using StaticArrays: SVector

const SupportedLane = Union{
    Float16, BFloat16, Float32, Float64,
    Float8_E5M2, Float8_E4M3, Float8_E3M4, Float8_E4M3FN, Float8_E8M0FNU,
    Float6_E2M3FN, Float6_E3M2FN, Float4_E2M1FN,
}
const SupportedMicrofloat = Union{
    Float8_E5M2, Float8_E4M3, Float8_E3M4, Float8_E4M3FN, Float8_E8M0FNU,
    Float6_E2M3FN, Float6_E3M2FN, Float4_E2M1FN,
}
const SourceVector2 = Union{SVector{2,<:SupportedLane},NVector{Float4_E2M1FN,2}}
const SourceVector4 = Union{SVector{4,<:SupportedLane},NVector{Float4_E2M1FN,4}}
const Vec2F16 = NTuple{2,VecElement{Float16}}

@generated function nvvm_f32x2_to_x2(::Val{name}, hi::Float32, lo::Float32) where name
    intr = String(name)
    ir = """
        declare i16 @$intr(float, float)

        define i16 @entry(float %hi, float %lo) {
        entry:
            %ret = call i16 @$intr(float %hi, float %lo)
            ret i16 %ret
        }
    """
    return :(Base.llvmcall(($ir, "entry"), UInt16, Tuple{Float32,Float32}, hi, lo))
end

@generated function nvvm_f16x2_to_x2(::Val{name}, xs::Vec2F16) where name
    intr = String(name)
    ir = """
        declare i16 @$intr(<2 x half>)

        define i16 @entry(<2 x half> %xs) {
        entry:
            %ret = call i16 @$intr(<2 x half> %xs)
            ret i16 %ret
        }
    """
    return :(Base.llvmcall(($ir, "entry"), UInt16, Tuple{Vec2F16}, xs))
end

@inline f16x2(lo::Float16, hi::Float16) = (VecElement(lo), VecElement(hi))

@inline nvvm_e4m3x2_rn(hi::Float32, lo::Float32) =
    nvvm_f32x2_to_x2(Val(Symbol("llvm.nvvm.ff.to.e4m3x2.rn")), hi, lo)
@inline nvvm_e5m2x2_rn(hi::Float32, lo::Float32) =
    nvvm_f32x2_to_x2(Val(Symbol("llvm.nvvm.ff.to.e5m2x2.rn")), hi, lo)
@inline nvvm_e4m3x2_rn(hi::Float16, lo::Float16) =
    nvvm_f16x2_to_x2(Val(Symbol("llvm.nvvm.f16x2.to.e4m3x2.rn")), f16x2(lo, hi))
@inline nvvm_e5m2x2_rn(hi::Float16, lo::Float16) =
    nvvm_f16x2_to_x2(Val(Symbol("llvm.nvvm.f16x2.to.e5m2x2.rn")), f16x2(lo, hi))
@inline nvvm_e2m3x2_rn_satfinite(hi::Float32, lo::Float32) =
    nvvm_f32x2_to_x2(Val(Symbol("llvm.nvvm.ff.to.e2m3x2.rn.satfinite")), hi, lo)
@inline nvvm_e3m2x2_rn_satfinite(hi::Float32, lo::Float32) =
    nvvm_f32x2_to_x2(Val(Symbol("llvm.nvvm.ff.to.e3m2x2.rn.satfinite")), hi, lo)
@inline nvvm_e2m1x2_rn_satfinite(hi::Float32, lo::Float32) =
    nvvm_f32x2_to_x2(Val(Symbol("llvm.nvvm.ff.to.e2m1x2.rn.satfinite")), hi, lo)
@inline nvvm_ue8m0x2_rz_satfinite(hi::Float32, lo::Float32) =
    nvvm_f32x2_to_x2(Val(Symbol("llvm.nvvm.ff.to.ue8m0x2.rz.satfinite")), hi, lo)

@inline function cc_ge(major::UInt32, minor::UInt32)
    cc = compute_capability()
    return cc.major > major || (cc.major == major && cc.minor >= minor)
end

@inline has_fp8_cvt() = cc_ge(0x00000008, 0x00000009)
@inline has_mxfp_cvt() = cc_ge(0x0000000a, 0x00000000) && target_feature_set() === :arch

@noinline device_throw_negative_unsigned() =
    CUDACore.@gputhrow "DomainError" "negative input to unsigned microfloat"
@noinline device_throw_no_nan() =
    CUDACore.@gputhrow "DomainError" "microfloat format has no NaN"
@noinline device_throw_no_overflow_sentinel() =
    CUDACore.@gputhrow "DomainError" "microfloat format has no overflow sentinel"

@inline device_clamp_floatmax(x::T) where T<:SupportedMicrofloat =
    signbit(x) ? -floatmax(T) : floatmax(T)
@inline device_clamp_inf(x::T) where T<:SupportedMicrofloat =
    signbit(x) ? -Microfloats.inf(T) : Microfloats.inf(T)

@inline function device_apply_overflow_policy(x::T, xf::Float32, mode::RoundingMode,
                                              ::Microfloats.Overflowing) where T<:SupportedMicrofloat
    if isnan(xf)
        Microfloats.hasnan(T) && return Microfloats.nan(T)
        device_throw_no_nan()
    elseif isinf(xf) || Microfloats.is_outside_floatmax(xf, T)
        if Microfloats.mode_overflows_to_inf(mode, signbit(xf))
            Microfloats.hasinf(T) && return device_clamp_inf(x)
            Microfloats.hasnan(T) && return Microfloats.nan(T)
            device_throw_no_overflow_sentinel()
        else
            return device_clamp_floatmax(x)
        end
    else
        return x
    end
end

@inline function device_apply_overflow_policy(x::T, xf::Float32, ::RoundingMode,
                                              ::Microfloats.Saturating) where T<:SupportedMicrofloat
    if isnan(xf)
        Microfloats.hasnan(T) && return Microfloats.nan(T)
        device_throw_no_nan()
    elseif isinf(xf) || Microfloats.is_outside_floatmax(xf, T)
        return device_clamp_floatmax(x)
    else
        return x
    end
end

@inline function device_round_to_microfloat(::Type{T}, x::Float32, rshift::F,
                                            mode::RoundingMode,
                                            policy::OverflowPolicy) where {T<:SupportedMicrofloat,F}
    if Microfloats.sign_bits(T) == 0 && signbit(x)
        device_throw_negative_unsigned()
    end
    iszero(x) && return signbit(x) ? -zero(T) : zero(T)

    f32_raw  = reinterpret(UInt32, x)
    f32_exp  = Int((f32_raw >> 23) & UInt32(0x000000ff))
    f32_frac = f32_raw & UInt32(0x007fffff)

    sig24 = f32_exp == 0 ? f32_frac : (UInt32(0x00800000) | f32_frac)
    true_exp = f32_exp == 0 ? -126 : (f32_exp - 127)
    t_exp = true_exp + Microfloats.exponent_bias(T)

    if t_exp <= 0
        shift = t_exp + Microfloats.significand_bits(T) - 24
        sub_q = rshift(sig24, -shift)
        max_frac = UInt32((1 << Microfloats.significand_bits(T)) - 1)
        if sub_q == 0
            t_raw = 0x00
        elseif sub_q == (UInt32(1) << Microfloats.significand_bits(T))
            t_raw = UInt8(1) << Microfloats.significand_bits(T)
        else
            sub_q = min(sub_q, max_frac)
            t_raw = UInt8(sub_q & max_frac)
        end
    else
        shift = 23 - Microfloats.significand_bits(T)
        total = rshift(sig24, shift)
        if total == 0
            t_raw = 0x00
        else
            t_exp_rounded = t_exp + Int(total >> (Microfloats.significand_bits(T) + 1))
            max_exp = (1 << Microfloats.exponent_bits(T)) - 1
            if t_exp_rounded > max_exp
                t_exp_rounded = max_exp
                if !Microfloats.hasinf(T)
                    total = (UInt32(1) << Microfloats.significand_bits(T)) |
                            UInt32((1 << Microfloats.significand_bits(T)) - 1)
                end
            end
            frac_field = UInt8(total) & UInt8((1 << Microfloats.significand_bits(T)) - 1)
            t_raw = (UInt8(t_exp_rounded) << Microfloats.significand_bits(T)) | frac_field
        end
    end

    t_raw |= (((f32_raw >> 31) % UInt8) <<
              (Microfloats.exponent_bits(T) + Microfloats.significand_bits(T))) &
             Microfloats.sign_mask(T)

    return device_apply_overflow_policy(reinterpret(T, t_raw), x, mode, policy)
end

@inline fallback_rn(::Type{T}, x::Float32, mode::RoundingMode{:Nearest},
                    overflow::OverflowPolicy) where T<:SupportedMicrofloat =
    device_round_to_microfloat(T, x, Microfloats.rshift_round_to_even, mode, overflow)

@inline fallback_rz(::Type{T}, x::Float32, mode::RoundingMode{:ToZero},
                    overflow::OverflowPolicy) where T<:SupportedMicrofloat =
    device_round_to_microfloat(T, x, Microfloats.rshift_truncate, mode, overflow)

@inline scalar_rn(::Type{T}, x::Float32, mode::RoundingMode{:Nearest},
                  overflow::OverflowPolicy) where T<:SupportedMicrofloat =
    fallback_rn(T, x, mode, overflow)

@inline scalar_rn(::Type{Float8_E4M3FN}, x::Float32, mode::RoundingMode{:Nearest},
                  overflow::OverflowPolicy) =
    has_fp8_cvt() && overflow === OVF ?
    reinterpret(Float8_E4M3FN, fp8_scalar(Float8_E4M3FN, x, overflow)) :
    fallback_rn(Float8_E4M3FN, x, mode, overflow)

@inline scalar_rn(::Type{Float8_E5M2}, x::Float32, mode::RoundingMode{:Nearest},
                  overflow::OverflowPolicy) =
    has_fp8_cvt() && overflow === OVF ?
    reinterpret(Float8_E5M2, fp8_scalar(Float8_E5M2, x, overflow)) :
    fallback_rn(Float8_E5M2, x, mode, overflow)

@inline scalar_rn(::Type{Float6_E2M3FN}, x::Float32, mode::RoundingMode{:Nearest},
                  overflow::OverflowPolicy) =
    has_mxfp_cvt() && overflow === SAT ?
    reinterpret(Float6_E2M3FN, UInt8(nvvm_e2m3x2_rn_satfinite(x, x) & 0x003f)) :
    fallback_rn(Float6_E2M3FN, x, mode, overflow)

@inline scalar_rn(::Type{Float6_E3M2FN}, x::Float32, mode::RoundingMode{:Nearest},
                  overflow::OverflowPolicy) =
    has_mxfp_cvt() && overflow === SAT ?
    reinterpret(Float6_E3M2FN, UInt8(nvvm_e3m2x2_rn_satfinite(x, x) & 0x003f)) :
    fallback_rn(Float6_E3M2FN, x, mode, overflow)

@inline scalar_rn(::Type{Float4_E2M1FN}, x::Float32, mode::RoundingMode{:Nearest},
                  overflow::OverflowPolicy) =
    has_mxfp_cvt() && overflow === SAT ?
    reinterpret(Float4_E2M1FN, UInt8(nvvm_e2m1x2_rn_satfinite(x, x) & 0x000f)) :
    fallback_rn(Float4_E2M1FN, x, mode, overflow)

@inline scalar_rz(::Type{T}, x::Float32, mode::RoundingMode{:ToZero},
                  overflow::OverflowPolicy) where T<:SupportedMicrofloat =
    fallback_rz(T, x, mode, overflow)

@inline scalar_rz(::Type{Float8_E8M0FNU}, x::Float32, mode::RoundingMode{:ToZero},
                  overflow::OverflowPolicy) =
    has_mxfp_cvt() && overflow === SAT && !signbit(x) ?
    reinterpret(Float8_E8M0FNU, UInt8(nvvm_ue8m0x2_rz_satfinite(x, x) & 0x00ff)) :
    fallback_rz(Float8_E8M0FNU, x, mode, overflow)

@inline lane32(x::SupportedLane) = Float32(x)

@inline nvec8x2(::Type{T}, raw::UInt16) where T =
    NArray{T,1,Tuple{2},UInt16}(raw)
@inline nvec8x4(::Type{T}, raw::UInt32) where T =
    NArray{T,1,Tuple{4},UInt32}(raw)
@inline nvec6x2(::Type{T}, raw::UInt16) where T =
    NArray{T,1,Tuple{2},NTuple{2,UInt8}}((UInt8(raw & 0x00ff),
                                           UInt8((raw >>> 8) & 0x00ff)))
@inline nvec6x4(::Type{T}, raw::UInt32) where T =
    NArray{T,1,Tuple{4},NTuple{3,UInt8}}((UInt8(raw & 0x000000ff),
                                           UInt8((raw >>> 8) & 0x000000ff),
                                           UInt8((raw >>> 16) & 0x000000ff)))
@inline nvec4x2(::Type{T}, raw::UInt8) where T =
    NArray{T,1,Tuple{2},UInt8}(raw)
@inline nvec4x4(::Type{T}, raw::UInt16) where T =
    NArray{T,1,Tuple{4},UInt16}(raw)

@inline byte_aligned_x2_to_u12(raw::UInt16) =
    (raw & 0x003f) | ((raw >>> 2) & 0x0fc0)

@inline fp8_scalar(::Type{Float8_E4M3FN}, x::Float32, overflow::OverflowPolicy) =
    UInt8((overflow === OVF ? nvvm_e4m3x2_rn(x, x) :
           0x0000) & 0x00ff)
@inline fp8_scalar(::Type{Float8_E5M2}, x::Float32, overflow::OverflowPolicy) =
    UInt8((overflow === OVF ? nvvm_e5m2x2_rn(x, x) :
           0x0000) & 0x00ff)

@inline function fp8x2_native_bits(::Type{Float8_E4M3FN}, lo::Float32, hi::Float32,
                                   overflow::OverflowPolicy)
    overflow === OVF && return nvvm_e4m3x2_rn(hi, lo)
    return nothing
end
@inline function fp8x2_native_bits(::Type{Float8_E4M3FN}, lo::Float16, hi::Float16,
                                   overflow::OverflowPolicy)
    overflow === OVF && return nvvm_e4m3x2_rn(hi, lo)
    return nothing
end

@inline function fp8x2_native_bits(::Type{Float8_E5M2}, lo::Float32, hi::Float32,
                                   overflow::OverflowPolicy)
    overflow === OVF && return nvvm_e5m2x2_rn(hi, lo)
    return nothing
end
@inline function fp8x2_native_bits(::Type{Float8_E5M2}, lo::Float16, hi::Float16,
                                   overflow::OverflowPolicy)
    overflow === OVF && return nvvm_e5m2x2_rn(hi, lo)
    return nothing
end

@inline fp8x2_native_bits_lanes(::Type{T}, lo::Float16, hi::Float16,
                                overflow::OverflowPolicy) where T =
    fp8x2_native_bits(T, lo, hi, overflow)
@inline fp8x2_native_bits_lanes(::Type{T}, lo::SupportedLane, hi::SupportedLane,
                                overflow::OverflowPolicy) where T =
    fp8x2_native_bits(T, lane32(lo), lane32(hi), overflow)

@inline fallback_x2(::Type{T}, xs::SourceVector2,
                    overflow::OverflowPolicy) where T<:SupportedMicrofloat =
    NArray{T,1,Tuple{2}}((T(lane32(xs[1]), RoundNearest; overflow=overflow),
                          T(lane32(xs[2]), RoundNearest; overflow=overflow)))

@inline fallback_x4(::Type{T}, xs::SourceVector4,
                    overflow::OverflowPolicy) where T<:SupportedMicrofloat =
    NArray{T,1,Tuple{4}}((T(lane32(xs[1]), RoundNearest; overflow=overflow),
                          T(lane32(xs[2]), RoundNearest; overflow=overflow),
                          T(lane32(xs[3]), RoundNearest; overflow=overflow),
                          T(lane32(xs[4]), RoundNearest; overflow=overflow)))

@device_override (::Type{T})(x::SupportedLane;
                             overflow::OverflowPolicy = overflow_policy(T)) where T<:SupportedMicrofloat =
    scalar_rn(T, lane32(x), RoundNearest, overflow)

@device_override (::Type{T})(x::SupportedLane, mode::RoundingMode{:Nearest};
                             overflow::OverflowPolicy = overflow_policy(T)) where T<:SupportedMicrofloat =
    scalar_rn(T, lane32(x), mode, overflow)

@device_override (::Type{T})(x::SupportedLane, mode::RoundingMode{:ToZero};
                             overflow::OverflowPolicy = overflow_policy(T)) where T<:SupportedMicrofloat =
    scalar_rz(T, lane32(x), mode, overflow)

@inline function fp8x2(::Type{T}, xs::SourceVector2,
                       overflow::OverflowPolicy) where T<:Union{Float8_E4M3FN,Float8_E5M2}
    if has_fp8_cvt()
        raw = fp8x2_native_bits_lanes(T, xs[1], xs[2], overflow)
        raw === nothing || return nvec8x2(T, raw)
    end
    return fallback_x2(T, xs, overflow)
end

@inline function fp8x4(::Type{T}, xs::SourceVector4,
                       overflow::OverflowPolicy) where T<:Union{Float8_E4M3FN,Float8_E5M2}
    if has_fp8_cvt()
        raw12 = fp8x2_native_bits_lanes(T, xs[1], xs[2], overflow)
        raw34 = fp8x2_native_bits_lanes(T, xs[3], xs[4], overflow)
        if raw12 !== nothing && raw34 !== nothing
            return nvec8x4(T, UInt32(raw12) | (UInt32(raw34) << 16))
        end
    end
    return fallback_x4(T, xs, overflow)
end

@inline function fp6x2(::Type{T}, xs::SourceVector2,
                       overflow::OverflowPolicy) where T<:Union{Float6_E2M3FN,Float6_E3M2FN}
    lo = lane32(xs[1])
    hi = lane32(xs[2])
    if has_mxfp_cvt() && overflow === SAT
        raw = T === Float6_E2M3FN ? nvvm_e2m3x2_rn_satfinite(hi, lo) :
                                    nvvm_e3m2x2_rn_satfinite(hi, lo)
        return nvec6x2(T, byte_aligned_x2_to_u12(raw))
    end
    return fallback_x2(T, xs, overflow)
end

@inline function fp6x4(::Type{T}, xs::SourceVector4,
                       overflow::OverflowPolicy) where T<:Union{Float6_E2M3FN,Float6_E3M2FN}
    x1 = lane32(xs[1])
    x2 = lane32(xs[2])
    x3 = lane32(xs[3])
    x4 = lane32(xs[4])
    if has_mxfp_cvt() && overflow === SAT
        raw12 = T === Float6_E2M3FN ? nvvm_e2m3x2_rn_satfinite(x2, x1) :
                                      nvvm_e3m2x2_rn_satfinite(x2, x1)
        raw34 = T === Float6_E2M3FN ? nvvm_e2m3x2_rn_satfinite(x4, x3) :
                                      nvvm_e3m2x2_rn_satfinite(x4, x3)
        bits12 = byte_aligned_x2_to_u12(raw12)
        bits34 = byte_aligned_x2_to_u12(raw34)
        return nvec6x4(T, UInt32(bits12) | (UInt32(bits34) << 12))
    end
    return fallback_x4(T, xs, overflow)
end

@inline function fp4x2(xs::SourceVector2, overflow::OverflowPolicy)
    lo = lane32(xs[1])
    hi = lane32(xs[2])
    if has_mxfp_cvt() && overflow === SAT
        return nvec4x2(Float4_E2M1FN, UInt8(nvvm_e2m1x2_rn_satfinite(hi, lo) & 0x00ff))
    end
    return fallback_x2(Float4_E2M1FN, xs, overflow)
end

@inline function fp4x4(xs::SourceVector4, overflow::OverflowPolicy)
    x1 = lane32(xs[1])
    x2 = lane32(xs[2])
    x3 = lane32(xs[3])
    x4 = lane32(xs[4])
    if has_mxfp_cvt() && overflow === SAT
        raw12 = UInt16(nvvm_e2m1x2_rn_satfinite(x2, x1) & 0x00ff)
        raw34 = UInt16(nvvm_e2m1x2_rn_satfinite(x4, x3) & 0x00ff)
        return nvec4x4(Float4_E2M1FN, raw12 | (raw34 << 8))
    end
    return fallback_x4(Float4_E2M1FN, xs, overflow)
end

@device_override (::Type{Float8x2_E4M3FN})(xs::SourceVector2;
                                           overflow::OverflowPolicy = overflow_policy(Float8_E4M3FN)) =
    fp8x2(Float8_E4M3FN, xs, overflow)
@device_override (::Type{Float8x2_E5M2})(xs::SourceVector2;
                                         overflow::OverflowPolicy = overflow_policy(Float8_E5M2)) =
    fp8x2(Float8_E5M2, xs, overflow)
@device_override (::Type{Float8x2_E8M0FNU})(xs::SourceVector2;
                                            overflow::OverflowPolicy = overflow_policy(Float8_E8M0FNU)) =
    fallback_x2(Float8_E8M0FNU, xs, overflow)

@device_override (::Type{Float8x4_E4M3FN})(xs::SourceVector4;
                                           overflow::OverflowPolicy = overflow_policy(Float8_E4M3FN)) =
    fp8x4(Float8_E4M3FN, xs, overflow)
@device_override (::Type{Float8x4_E5M2})(xs::SourceVector4;
                                         overflow::OverflowPolicy = overflow_policy(Float8_E5M2)) =
    fp8x4(Float8_E5M2, xs, overflow)
@device_override (::Type{Float8x4_E8M0FNU})(xs::SourceVector4;
                                            overflow::OverflowPolicy = overflow_policy(Float8_E8M0FNU)) =
    fallback_x4(Float8_E8M0FNU, xs, overflow)

@device_override (::Type{Float6x2_E2M3FN})(xs::SourceVector2;
                                           overflow::OverflowPolicy = overflow_policy(Float6_E2M3FN)) =
    fp6x2(Float6_E2M3FN, xs, overflow)
@device_override (::Type{Float6x2_E3M2FN})(xs::SourceVector2;
                                           overflow::OverflowPolicy = overflow_policy(Float6_E3M2FN)) =
    fp6x2(Float6_E3M2FN, xs, overflow)
@device_override (::Type{Float4x2_E2M1FN})(xs::SourceVector2;
                                           overflow::OverflowPolicy = overflow_policy(Float4_E2M1FN)) =
    fp4x2(xs, overflow)

@device_override (::Type{Float6x4_E2M3FN})(xs::SourceVector4;
                                           overflow::OverflowPolicy = overflow_policy(Float6_E2M3FN)) =
    fp6x4(Float6_E2M3FN, xs, overflow)
@device_override (::Type{Float6x4_E3M2FN})(xs::SourceVector4;
                                           overflow::OverflowPolicy = overflow_policy(Float6_E3M2FN)) =
    fp6x4(Float6_E3M2FN, xs, overflow)
@device_override (::Type{Float4x4_E2M1FN})(xs::SourceVector4;
                                           overflow::OverflowPolicy = overflow_policy(Float4_E2M1FN)) =
    fp4x4(xs, overflow)

end
