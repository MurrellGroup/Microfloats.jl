# Device-side conversion overrides.
#
# Microfloats funnels every conversion through a few dispatchable methods:
#
#   cvt(T, x, mode, policy)                  scalar narrowing
#   cvt(SVector{N,T}, xs, mode, policy)      N lanes, one byte per lane
#   cvt(NVector{T,N}, xs, mode, policy)      N lanes, densely packed
#   cvt(F, x)                                scalar widening
#   cvt(SVector{N,F}, xs)                    N lanes widened
#
# The packed narrowing default converts through the `SVector` form and packs,
# and packed widening sources unpack into a tuple of lanes. Both are layout
# only and fold away, so this file specializes the unpacked forms alone and
# every container gets the native instruction.
#
# It overrides:
#
#   1. the error hooks, so the *generic* numeric kernels run unmodified on
#      device (no duplicated conversion body);
#   2. narrow `cvt` signatures for exactly the combinations that have a native
#      PTX conversion instruction, each gated on what the compile target
#      offers. The gates fold at kernel compile time because
#      `compute_capability()`, the target's feature set and
#      `ptx_isa_version()` are compile-time constants under GPUCompiler, so each kernel compiles
#      to either the native instruction or the generic path, with no runtime
#      branch;
#   3. `max_twiddle_cost`, so `@cvt_table` methods prefer a lookup on device
#      where the host prefers a bit-twiddle.
#
# PTX `cvt` into a sub-byte or 8-bit float format only exists as `.satfinite`,
# so every native narrowing implements the `SAT` overflow policy; `OVF`
# always takes the generic path. Widening is exact and needs no policy.
#
# Operand order follows PTX ISA "cvt": `cvt d, a, b` puts `a` in the UPPER
# half of `d` and `b` in the LOWER half, and packed 16-bit sources convert
# upper half to upper half. Lane 1 is always the low half here.
#
# Native and generic results agree bit for bit over every representable value,
# every tie and the saturating range; `test/cuda_extension.jl` sweeps this on
# the device. NaN payloads are the one hardware-defined part.

using Microfloats
using Microfloats: Microfloat, cvt, cvt_generic, cvt_twiddle, cvt_lanes,
                   OverflowPolicy, Saturating, BFloat16, bitwidth,
                   throw_negative_unsigned, throw_no_nan,
                   Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU,
                   Float6_E2M3FN, Float6_E3M2FN, Float4_E2M1FN
using StaticArrays: SVector
using CUDACore: CUDACore, @device_override, compute_capability, ptx_isa_version

# ───────────────────────── error hooks ──────────────────────────

# With these five overrides the generic conversion kernels are device-safe
# as-is; everything below is optimization only.
# NB: the `where T` type variables are load-bearing — a bare `::Type`
# argument is left unspecialized by Julia, which turns these into dynamic
# calls in device code (InvalidIRError).
@device_override @noinline Microfloats.throw_negative_unsigned(::Type{T}, x) where T =
    CUDACore.@gputhrow "DomainError" "negative input to unsigned microfloat"
@device_override @noinline Microfloats.throw_negate_unsigned(::Type{T}, x) where T =
    CUDACore.@gputhrow "DomainError" "cannot negate unsigned microfloat"
@device_override @noinline Microfloats.throw_no_nan(::Type{T}, x) where T =
    CUDACore.@gputhrow "DomainError" "microfloat format has no NaN"
@device_override @noinline Microfloats.throw_no_overflow_sentinel(::Type{T}, x) where T =
    CUDACore.@gputhrow "DomainError" "microfloat format has no overflow sentinel; use overflow=SAT"
@device_override @noinline Microfloats.throw_unsupported_rounding(::Type{T}, mode) where T =
    CUDACore.@gputhrow "ArgumentError" "unsupported rounding mode for microfloat conversion"

# ───────────────────────── capability gates ──────────────────────────

@inline function at_least(v, major::Integer, minor::Integer)
    v.major > major || (v.major == major && v.minor >= minor)
end

# fp8 with Float32 and Float16 operands: every sm_89+ target.
@inline has_fp8() = at_least(compute_capability(), 8, 9)
# fp6, fp4 and ue8m0 with Float32 operands, and their Float16 widening: the
# arch- and family-specific sm_100+ targets, never a baseline target.
@inline has_mxfp() = at_least(compute_capability(), 10, 0) && !baseline_target()

# `target_feature_set()` returns a Symbol, and a comparison of symbols does
# not always fold in device code: the kernel then keeps both the native and
# the generic path behind a runtime branch. The integer it is derived from
# always folds.
const BASELINE_FEATURES = UInt32(isdefined(CUDACore, :BaselineFeatures) ?
    CUDACore.BaselineFeatures : CUDACore.GPUCompiler.BaselineFeatures)
@inline baseline_target() = CUDACore.sm_features() == BASELINE_FEATURES
# Float16 and BFloat16 operands for the remaining narrowing forms (PTX 9.1)
# and BFloat16 widening (PTX 9.2), on the same targets.
@inline has_mxfp_half() = has_mxfp() && has_ptx(Val(9), Val(1))
@inline has_bf16_widening() = has_mxfp() && has_ptx(Val(9), Val(2))

# The version a PTX module declares is the minimum its tools must support.
# GPUCompiler bounds it by what LLVM's NVPTX backend knows, which says nothing
# about inline asm: what decides whether an instruction assembles is ptxas.
# So a newer form is also taken when the assembler of this session knows it.
# The generator runs on the host while a kernel is compiled, where the
# toolchain is fixed.
@inline has_ptx(::Val{major}, ::Val{minor}) where {major,minor} =
    at_least(ptx_isa_version(), major, minor) || ptxas_has(Val(major), Val(minor))
@generated function ptxas_has(::Val{major}, ::Val{minor}) where {major,minor}
    try
        any(>=(VersionNumber(major, minor)), CUDACore.ptxas_compat().ptx)
    catch
        false
    end
end

# ───────────────────────── PTX cvt wrappers ──────────────────────────

# Inline asm rather than `llvm.nvvm.*` intrinsics, so availability doesn't
# depend on the LLVM version Julia ships. Three operand shapes cover every
# form: two Float32 lanes to a packed pair, a packed 16-bit pair to a packed
# narrow pair, and the reverse.
#
# fp4 pairs are `.b8`, which has no NVPTX register-constraint letter, so
# those forms bridge through a 16-bit register (as NVIDIA's <cuda_fp4.hpp>
# and PTX.jl do).

asm_f32(instr, nibbles) = nibbles ?
    "{ .reg .b8 t; $instr t, \$1, \$2; mov.b16 \$0, {t, 0}; }" : "$instr \$0, \$1, \$2;"
asm_narrow(instr, nibbles) = nibbles ?
    "{ .reg .b8 t; $instr t, \$1; mov.b16 \$0, {t, 0}; }" : "$instr \$0, \$1;"
asm_widen(instr, nibbles) = nibbles ?
    "{ .reg .b8 t, hi; mov.b16 {t, hi}, \$1; $instr \$0, t; }" : "$instr \$0, \$1;"

@generated function ptx_cvt_f32(::Val{instr}, ::Val{nibbles}, lo::Float32, hi::Float32) where {instr,nibbles}
    ir = """
        define i16 @entry(float %lo, float %hi) #0 {
            %r = call i16 asm "$(asm_f32(String(instr), nibbles))", "=h,f,f"(float %hi, float %lo)
            ret i16 %r
        }
        attributes #0 = { alwaysinline }
    """
    :(Base.llvmcall(($ir, "entry"), UInt16, Tuple{Float32,Float32}, lo, hi))
end

@generated function ptx_cvt_narrow(::Val{instr}, ::Val{nibbles}, pair::UInt32) where {instr,nibbles}
    ir = """
        define i16 @entry(i32 %a) #0 {
            %r = call i16 asm "$(asm_narrow(String(instr), nibbles))", "=h,r"(i32 %a)
            ret i16 %r
        }
        attributes #0 = { alwaysinline }
    """
    :(Base.llvmcall(($ir, "entry"), UInt16, Tuple{UInt32}, pair))
end

@generated function ptx_cvt_widen(::Val{instr}, ::Val{nibbles}, pair::UInt16) where {instr,nibbles}
    ir = """
        define i32 @entry(i16 %a) #0 {
            %r = call i32 asm "$(asm_widen(String(instr), nibbles))", "=r,h"(i16 %a)
            ret i32 %r
        }
        attributes #0 = { alwaysinline }
    """
    :(Base.llvmcall(($ir, "entry"), UInt32, Tuple{UInt16}, pair))
end

# ───────────────────────── register layouts ──────────────────────────

# A narrow pair register holds lane 1 in its low half. fp8, fp6 and ue8m0
# pairs take one byte per lane, which is `SVector{2,T}`; an fp4 pair takes one
# nibble per lane in a single byte, which is `NVector{T,2}`.

@inline narrow_lanes(::Type{T}, ::Val{false}, bits::UInt16) where T =
    (reinterpret(T, bits % UInt8), reinterpret(T, (bits >> 8) % UInt8))
@inline narrow_lanes(::Type{T}, ::Val{true}, bits::UInt16) where T =
    (reinterpret(T, (bits % UInt8) & 0x0f), reinterpret(T, (bits % UInt8) >> 4))

# Sources may carry set bits above the format's width (they are ignored by
# every Microfloats operation), while the hardware requires them to be zero.
@inline lane_bits(x::T) where T<:Microfloat =
    reinterpret(UInt8, x) & (0xff >> (8 - bitwidth(T)))
@inline narrow_pair(::Val{false}, lo::Microfloat, hi::Microfloat) =
    UInt16(lane_bits(lo)) | (UInt16(lane_bits(hi)) << 8)
@inline narrow_pair(::Val{true}, lo::Microfloat, hi::Microfloat) =
    UInt16(lane_bits(lo) | (lane_bits(hi) << 4))

const Half = Union{Float16,BFloat16}
@inline half_pair(lo::H, hi::H) where H<:Half =
    UInt32(reinterpret(UInt16, lo)) | (UInt32(reinterpret(UInt16, hi)) << 16)
@inline half_lanes(::Type{H}, bits::UInt32) where H<:Half =
    (reinterpret(H, bits % UInt16), reinterpret(H, (bits >> 16) % UInt16))

# One native pair conversion, as a callable so the lane loops below stay
# generic: `Narrow` maps two wide lanes to two `T` lanes, `Widen` the reverse.
struct Narrow{T,instr,nibbles} end
@inline (::Narrow{T,instr,nibbles})(lo::Float32, hi::Float32) where {T,instr,nibbles} =
    narrow_lanes(T, Val(nibbles), ptx_cvt_f32(Val(instr), Val(nibbles), lo, hi))
@inline (::Narrow{T,instr,nibbles})(lo::H, hi::H) where {T,instr,nibbles,H<:Half} =
    narrow_lanes(T, Val(nibbles), ptx_cvt_narrow(Val(instr), Val(nibbles), half_pair(lo, hi)))

struct Widen{H,instr,nibbles} end
@inline (::Widen{H,instr,nibbles})(lo::Microfloat, hi::Microfloat) where {H,instr,nibbles} =
    half_lanes(H, ptx_cvt_widen(Val(instr), Val(nibbles), narrow_pair(Val(nibbles), lo, hi)))

# N lanes, two at a time. Every pair is converted exactly once: inline asm is
# opaque to LLVM, so a repeated call would not be merged.
@inline function by_pairs(op, xs::NTuple{N,Any}) where N
    pairs = ntuple(j -> op(xs[2j - 1], xs[2j]), Val(N ÷ 2))
    ntuple(i -> pairs[(i + 1) >> 1][2 - (i & 1)], Val(N))
end

# ───────────────────────── narrowing ──────────────────────────

# The generic path throws where the hardware would silently produce a value:
# NaN into a format without NaN, negative input into an unsigned format.
@inline guard(::Val{:none}, ::Type{T}, xs) where T = nothing
@inline guard(::Val{:nan}, ::Type{T}, xs) where T =
    (any(isnan, xs) && throw_no_nan(T, xs); nothing)
@inline guard(::Val{:sign}, ::Type{T}, xs) where T =
    (any(signbit, xs) && throw_negative_unsigned(T, xs); nothing)

# A source without a native form still reaches the Float32 native.
@inline scalar_fallback(::Type{T}, x::Float32, mode, policy) where T = cvt_twiddle(T, x, mode, policy)
@inline scalar_fallback(::Type{T}, x, mode, policy) where T = cvt(T, Float32(x), mode, policy)

# (target, PTX type, nibble pairs, guard, modes, (source, PTX source, gate)...)
const RN = ((RoundingMode{:Nearest}, "rn"),)
const NARROWING = (
    (Float8_E4M3FN,  "e4m3x2",  false, :none, RN,
        ((Float32, "f32", :has_fp8), (Float16, "f16x2", :has_fp8), (BFloat16, "bf16x2", :has_mxfp_half))),
    (Float8_E5M2,    "e5m2x2",  false, :none, RN,
        ((Float32, "f32", :has_fp8), (Float16, "f16x2", :has_fp8), (BFloat16, "bf16x2", :has_mxfp_half))),
    (Float6_E2M3FN,  "e2m3x2",  false, :nan,  RN,
        ((Float32, "f32", :has_mxfp), (Float16, "f16x2", :has_mxfp_half), (BFloat16, "bf16x2", :has_mxfp_half))),
    (Float6_E3M2FN,  "e3m2x2",  false, :nan,  RN,
        ((Float32, "f32", :has_mxfp), (Float16, "f16x2", :has_mxfp_half), (BFloat16, "bf16x2", :has_mxfp_half))),
    (Float4_E2M1FN,  "e2m1x2",  true,  :nan,  RN,
        ((Float32, "f32", :has_mxfp), (Float16, "f16x2", :has_mxfp_half), (BFloat16, "bf16x2", :has_mxfp_half))),
    # Hardware converts to ue8m0 only toward zero and upward; the type's
    # default RoundNearest keeps the generic path. NaN maps to 0xff natively,
    # which is `nan(Float8_E8M0FNU)`.
    (Float8_E8M0FNU, "ue8m0x2", false, :sign, ((RoundingMode{:ToZero}, "rz"), (RoundingMode{:Up}, "rp")),
        ((Float32, "f32", :has_mxfp), (BFloat16, "bf16x2", :has_mxfp))),
)

for (T, name, nibbles, check, modes, sources) in NARROWING,
    (M, rounding) in modes, (S, source, gate) in sources

    op = Narrow{T,Symbol("cvt.$rounding.satfinite.$name.$source"),nibbles}()
    g = Val(check)
    @eval begin
        @device_override @inline function Microfloats.cvt(::Type{$T}, x::$S, mode::$M, policy::Saturating)
            $gate() || return scalar_fallback($T, x, mode, policy)
            guard($g, $T, (x,))
            return $op(x, x)[1]
        end

        @device_override @inline function Microfloats.cvt(::Type{SVector{N,$T}}, xs::NTuple{N,$S},
                                                          mode::$M, policy::Saturating) where N
            ($gate() && iseven(N)) || return cvt_lanes(SVector{N,$T}, xs, mode, policy)
            guard($g, $T, xs)
            return SVector{N,$T}(by_pairs($op, xs))
        end
    end
end

# ───────────────────────── widening ──────────────────────────

# Widening is exact. NaN payloads and the sign of a widened NaN are
# hardware-defined; everything else matches the generic lookup.
#
# (source, PTX type, nibble pairs, (destination, PTX destination, gate)...)
const WIDENING = (
    (Float8_E4M3FN,  "e4m3x2",  false, ((Float16, "f16x2", :has_fp8),  (BFloat16, "bf16x2", :has_bf16_widening))),
    (Float8_E5M2,    "e5m2x2",  false, ((Float16, "f16x2", :has_fp8),  (BFloat16, "bf16x2", :has_bf16_widening))),
    (Float6_E2M3FN,  "e2m3x2",  false, ((Float16, "f16x2", :has_mxfp), (BFloat16, "bf16x2", :has_bf16_widening))),
    (Float6_E3M2FN,  "e3m2x2",  false, ((Float16, "f16x2", :has_mxfp), (BFloat16, "bf16x2", :has_bf16_widening))),
    (Float4_E2M1FN,  "e2m1x2",  true,  ((Float16, "f16x2", :has_mxfp), (BFloat16, "bf16x2", :has_bf16_widening))),
    (Float8_E8M0FNU, "ue8m0x2", false, ((BFloat16, "bf16x2", :has_mxfp),)),
)

for (T, name, nibbles, destinations) in WIDENING, (H, destination, gate) in destinations
    op = Widen{H,Symbol("cvt.rn.$destination.$name"),nibbles}()
    # Float32 has no direct form: widen to BFloat16 and extend, two register
    # instructions per lane instead of the generic path's table load.
    wide = H === BFloat16 ? (BFloat16, Float32) : (H,)
    for F in wide
        @eval begin
            @device_override @inline function Microfloats.cvt(::Type{$F}, x::$T)
                $gate() || return cvt_generic($F, x)
                return $F($op(x, x)[1])
            end

            @device_override @inline function Microfloats.cvt(::Type{SVector{N,$F}}, xs::NTuple{N,$T}) where N
                ($gate() && iseven(N)) || return cvt_lanes(SVector{N,$F}, xs)
                return SVector{N,$F}(map($F, by_pairs($op, xs)))
            end
        end
    end
end

# ───────────────────────── @cvt_table methods ──────────────────────────

# A lookup in a small constant table is one load through the read-only cache,
# which in device code beats a bit-twiddle of more than one linear piece and
# a sign move. A lone shift, such as Float4_E2M1FN to Float6_E2M3FN, still
# wins.
@device_override Microfloats.max_twiddle_cost() = 2
