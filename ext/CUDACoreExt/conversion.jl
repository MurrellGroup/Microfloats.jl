# Device-side conversion overrides.
#
# Microfloats funnels every conversion through `cvt(T, x, mode, policy)`
# (scalar), `cvt(SVector{N,T}, xs, mode, policy)` (one lane per byte) and
# `cvt(NVector{T,N}, xs, mode, policy)` (densely packed), with the rounding
# mode and overflow policy as positional, dispatchable arguments. The packed
# default converts through the SVector form, so an SVector override also
# serves dense destinations.
# This file therefore only needs to override:
#
#   1. the error hooks — so the *generic* numeric kernels run unmodified on
#      device (no duplicated conversion body);
#   2. narrow `cvt` signatures for exactly the (target, source, mode, policy)
#      combinations that have native PTX conversion instructions, gated on
#      compute capability. The gates fold at kernel compile time because
#      `compute_capability()`/`target_feature_set()` are compile-time
#      constants under GPUCompiler, so each kernel compiles to either the
#      native instruction or the generic path with no runtime branch.
#
# PTX `cvt` into sub-byte float formats is only available as `.satfinite`
# (mandatory for fp8/fp6/fp4/ue8m0 destinations), so every native path
# implements the `SAT` overflow policy; `OVF` always takes the generic path.
# Instruction/operand-order conventions follow PTX ISA §9.7.9 ("cvt") as
# validated empirically in PTX.jl (H100/GB10): `cvt d, a, b` puts `a` in the
# UPPER lane and `b` in the LOWER lane of `d`.
#
# EXPERIMENTAL: exact parity between the native instructions and the generic
# path (rounding at the floatmax boundary, NaN payloads) has not been
# validated on hardware yet; run an on-device exhaustive parity sweep before
# relying on bit-exactness.

using Microfloats
using Microfloats: Microfloat, cvt, cvt_generic, cvt_lanes,
                   OverflowPolicy, Overflowing, Saturating, SAT, OVF,
                   throw_negative_unsigned, throw_no_nan,
                   Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU,
                   Float6_E2M3FN, Float6_E3M2FN, Float4_E2M1FN
using BitPacking: NArray, NVector
using StaticArrays: SVector
using CUDACore: CUDACore, @device_override, compute_capability, target_feature_set

# ───────────────────────── error hooks ──────────────────────────

# With these four overrides the generic conversion kernels are device-safe
# as-is; everything below is optimization only.
# NB: the `where T` type variables are load-bearing — a bare `::Type`
# argument is left unspecialized by Julia, which turns these into dynamic
# calls in device code (InvalidIRError).
@device_override @noinline Microfloats.throw_negative_unsigned(::Type{T}, x) where T =
    CUDACore.@gputhrow "DomainError" "negative input to unsigned microfloat"
@device_override @noinline Microfloats.throw_no_nan(::Type{T}, x) where T =
    CUDACore.@gputhrow "DomainError" "microfloat format has no NaN"
@device_override @noinline Microfloats.throw_no_overflow_sentinel(::Type{T}, x) where T =
    CUDACore.@gputhrow "DomainError" "microfloat format has no overflow sentinel; use overflow=SAT"
@device_override @noinline Microfloats.throw_unsupported_rounding(::Type{T}, mode) where T =
    CUDACore.@gputhrow "ArgumentError" "unsupported rounding mode for microfloat conversion"

# ───────────────────────── capability gates ──────────────────────────

@inline function cc_ge(major::UInt32, minor::UInt32)
    cc = compute_capability()
    cc.major > major || (cc.major == major && cc.minor >= minor)
end

@inline has_fp8_cvt() = cc_ge(UInt32(8), UInt32(9))
@inline has_mxfp_cvt() = cc_ge(UInt32(10), UInt32(0)) && target_feature_set() === :arch

# ───────────────────────── PTX cvt wrappers ──────────────────────────

# Two Float32 lanes → one packed pair, low lane first. Inline asm rather
# than `llvm.nvvm.*` intrinsics so availability doesn't depend on the LLVM
# version Julia ships; a future PTX.jl-based extension can supersede these
# with intrinsic-backed lowering.
@generated function cvt_pair_bits(::Val{instr}, lo::Float32, hi::Float32) where instr
    ir = """
        define i16 @entry(float %lo, float %hi) #0 {
            %r = call i16 asm "$(String(instr)) \$0, \$1, \$2;", "=h,f,f"(float %hi, float %lo)
            ret i16 %r
        }
        attributes #0 = { alwaysinline }
    """
    :(Base.llvmcall(($ir, "entry"), UInt16, Tuple{Float32,Float32}, lo, hi))
end

# fp4 destinations are `.b8`, which has no NVPTX register-constraint letter;
# bridge through a 16-bit register (mirrors NVIDIA's <cuda_fp4.hpp> shims and
# PTX.jl's hand-written e2m1x2 entries).
@generated function cvt_pair_bits_b8(::Val{instr}, lo::Float32, hi::Float32) where instr
    ir = """
        define i16 @entry(float %lo, float %hi) #0 {
            %r = call i16 asm "{ .reg .b8 t; $(String(instr)) t, \$1, \$2; mov.b16 \$0, {t, 0}; }", "=h,f,f"(float %hi, float %lo)
            ret i16 %r
        }
        attributes #0 = { alwaysinline }
    """
    :(Base.llvmcall(($ir, "entry"), UInt16, Tuple{Float32,Float32}, lo, hi))
end

# ───────────────────────── result layouts ──────────────────────────

# PTX fp8/fp6/ue8m0 pair conversions return one lane per byte, lane 1 (`b`)
# in the low byte: an SVector of one-byte Microfloats. For 8-bit formats that
# is also the dense NVector layout. fp4 pairs come back dense, two lanes per
# byte, which is the NVector layout.

@inline byte_lanes(::Type{T}, bits::UInt16) where {T} =
    SVector{2,T}(reinterpret(T, bits % UInt8), reinterpret(T, (bits >> 8) % UInt8))

@inline pack2(::Type{T}, data::D) where {T,D} = NArray{T,1,Tuple{2},D}(data)
@inline pack4(::Type{T}, data::D) where {T,D} = NArray{T,1,Tuple{4},D}(data)

# ───────────────────────── fp8: E4M3FN / E5M2 (sm_89+) ──────────────────────────

for (T, instr) in ((Float8_E4M3FN, "cvt.rn.satfinite.e4m3x2.f32"),
                   (Float8_E5M2,   "cvt.rn.satfinite.e5m2x2.f32"))
    v = Val(Symbol(instr))
    @eval begin
        @device_override @inline Microfloats.cvt(::Type{$T}, x::Float32,
                                                 mode::RoundingMode{:Nearest}, policy::Saturating) =
            has_fp8_cvt() ? reinterpret($T, cvt_pair_bits($v, x, x) % UInt8) :
                            cvt_generic($T, x, mode, policy)

        @device_override @inline Microfloats.cvt(::Type{NVector{$T,2}}, xs::NTuple{2,Float32},
                                                 mode::RoundingMode{:Nearest}, policy::Saturating) =
            has_fp8_cvt() ? pack2($T, cvt_pair_bits($v, xs[1], xs[2])) :
                            cvt_lanes(NVector{$T,2}, xs, mode, policy)

        @device_override @inline Microfloats.cvt(::Type{NVector{$T,4}}, xs::NTuple{4,Float32},
                                                 mode::RoundingMode{:Nearest}, policy::Saturating) =
            has_fp8_cvt() ? pack4($T, UInt32(cvt_pair_bits($v, xs[1], xs[2])) |
                                      (UInt32(cvt_pair_bits($v, xs[3], xs[4])) << 16)) :
                            cvt_lanes(NVector{$T,4}, xs, mode, policy)
    end
end

# ───────────────────────── fp6: E2M3FN / E3M2FN (sm_100a+) ──────────────────────────

# FiniteOnly targets: the generic SAT path throws for NaN inputs, while
# hardware `.satfinite` silently maps NaN; guard first to keep semantics.
for (T, instr) in ((Float6_E2M3FN, "cvt.rn.satfinite.e2m3x2.f32"),
                   (Float6_E3M2FN, "cvt.rn.satfinite.e3m2x2.f32"))
    v = Val(Symbol(instr))
    @eval begin
        @device_override @inline function Microfloats.cvt(::Type{$T}, x::Float32,
                                                          mode::RoundingMode{:Nearest}, policy::Saturating)
            has_mxfp_cvt() || return cvt_generic($T, x, mode, policy)
            isnan(x) && throw_no_nan($T, x)
            return reinterpret($T, (cvt_pair_bits($v, x, x) % UInt8) & 0x3f)
        end

        # One lane per byte is the native result, so no repacking. Dense
        # NVector{T,N} destinations pack this result through the generic funnel.
        @device_override @inline function Microfloats.cvt(::Type{SVector{2,$T}}, xs::NTuple{2,Float32},
                                                          mode::RoundingMode{:Nearest}, policy::Saturating)
            has_mxfp_cvt() || return cvt_lanes(SVector{2,$T}, xs, mode, policy)
            (isnan(xs[1]) | isnan(xs[2])) && throw_no_nan($T, xs)
            return byte_lanes($T, cvt_pair_bits($v, xs[1], xs[2]))
        end

        @device_override @inline function Microfloats.cvt(::Type{SVector{4,$T}}, xs::NTuple{4,Float32},
                                                          mode::RoundingMode{:Nearest}, policy::Saturating)
            has_mxfp_cvt() || return cvt_lanes(SVector{4,$T}, xs, mode, policy)
            (isnan(xs[1]) | isnan(xs[2]) | isnan(xs[3]) | isnan(xs[4])) && throw_no_nan($T, xs)
            lo = byte_lanes($T, cvt_pair_bits($v, xs[1], xs[2]))
            hi = byte_lanes($T, cvt_pair_bits($v, xs[3], xs[4]))
            return SVector{4,$T}(lo[1], lo[2], hi[1], hi[2])
        end
    end
end

# ───────────────────────── fp4: E2M1FN (sm_100a+) ──────────────────────────

let T = Float4_E2M1FN, v = Val(Symbol("cvt.rn.satfinite.e2m1x2.f32"))
    @eval begin
        @device_override @inline function Microfloats.cvt(::Type{$T}, x::Float32,
                                                          mode::RoundingMode{:Nearest}, policy::Saturating)
            has_mxfp_cvt() || return cvt_generic($T, x, mode, policy)
            isnan(x) && throw_no_nan($T, x)
            return reinterpret($T, (cvt_pair_bits_b8($v, x, x) % UInt8) & 0x0f)
        end

        @device_override @inline function Microfloats.cvt(::Type{NVector{$T,2}}, xs::NTuple{2,Float32},
                                                          mode::RoundingMode{:Nearest}, policy::Saturating)
            has_mxfp_cvt() || return cvt_lanes(NVector{$T,2}, xs, mode, policy)
            (isnan(xs[1]) | isnan(xs[2])) && throw_no_nan($T, xs)
            return pack2($T, cvt_pair_bits_b8($v, xs[1], xs[2]) % UInt8)
        end

        @device_override @inline function Microfloats.cvt(::Type{NVector{$T,4}}, xs::NTuple{4,Float32},
                                                          mode::RoundingMode{:Nearest}, policy::Saturating)
            has_mxfp_cvt() || return cvt_lanes(NVector{$T,4}, xs, mode, policy)
            (isnan(xs[1]) | isnan(xs[2]) | isnan(xs[3]) | isnan(xs[4])) && throw_no_nan($T, xs)
            return pack4($T, (cvt_pair_bits_b8($v, xs[1], xs[2]) & 0x00ff) |
                             (cvt_pair_bits_b8($v, xs[3], xs[4]) << 8))
        end
    end
end

# ───────────────────────── ue8m0: E8M0FNU (sm_100a+) ──────────────────────────

# Hardware only converts to ue8m0 with .rz (and .rp); the type's default
# RoundNearest keeps the generic path. NaN maps to 0xff natively, matching
# `nan(Float8_E8M0FNU)` under SAT. The generic path throws for any negative
# input (including -0.0); guard to preserve that.
let T = Float8_E8M0FNU, v = Val(Symbol("cvt.rz.satfinite.ue8m0x2.f32"))
    @eval begin
        @device_override @inline function Microfloats.cvt(::Type{$T}, x::Float32,
                                                          mode::RoundingMode{:ToZero}, policy::Saturating)
            has_mxfp_cvt() || return cvt_generic($T, x, mode, policy)
            signbit(x) && throw_negative_unsigned($T, x)
            return reinterpret($T, cvt_pair_bits($v, x, x) % UInt8)
        end

        @device_override @inline function Microfloats.cvt(::Type{NVector{$T,2}}, xs::NTuple{2,Float32},
                                                          mode::RoundingMode{:ToZero}, policy::Saturating)
            has_mxfp_cvt() || return cvt_lanes(NVector{$T,2}, xs, mode, policy)
            (signbit(xs[1]) | signbit(xs[2])) && throw_negative_unsigned($T, xs)
            return pack2($T, cvt_pair_bits($v, xs[1], xs[2]))
        end

        @device_override @inline function Microfloats.cvt(::Type{NVector{$T,4}}, xs::NTuple{4,Float32},
                                                          mode::RoundingMode{:ToZero}, policy::Saturating)
            has_mxfp_cvt() || return cvt_lanes(NVector{$T,4}, xs, mode, policy)
            (signbit(xs[1]) | signbit(xs[2]) | signbit(xs[3]) | signbit(xs[4])) &&
                throw_negative_unsigned($T, xs)
            return pack4($T, UInt32(cvt_pair_bits($v, xs[1], xs[2])) |
                             (UInt32(cvt_pair_bits($v, xs[3], xs[4])) << 16))
        end
    end
end
