abstract type OverflowPolicy end

struct Overflowing <: OverflowPolicy end

struct Saturating <: OverflowPolicy end

"""
    OVF

Policy that maps numeric overflow to a sentinel — `±Inf` when the format
has infinity, otherwise `NaN` when it has NaN, otherwise a `DomainError`.

Pass directly as the value of `overflow=` in `@microfloat` declarations
or as a per-call keyword.

| Input                  | [`IEEE`](@ref) | [`NanOnlyAllOnes`](@ref) | [`FiniteOnly`](@ref) |
| ---------------------- | -------------- | ------------------------ | -------------------- |
| `isnan(x)`             | NaN            | NaN                      | Error                |
| `abs(x) > floatmax(T)` | ±Inf           | NaN                      | Error                |

The table above describes the *default-mode* (`RoundNearest`) behavior.
Directed modes like `RoundToZero` saturate per IEEE-754 regardless of
policy.

See also [`SAT`](@ref).

# Examples
```jldoctest
julia> @microfloat OverflowingFloat8 exponent=4 significand=3 overflow=Microfloats.OVF

julia> OverflowingFloat8(10000)
OverflowingFloat8(Inf)

julia> Float8_E4M3(10000; overflow=Microfloats.OVF)
Float8_E4M3(Inf)
```
"""
const OVF = Overflowing()

"""
    SAT

Policy that clamps numeric overflow to `±floatmax(T)`. NaN inputs pass
through if `T` has NaN, else throw a `DomainError`.

Pass directly as the value of `overflow=` in `@microfloat` declarations
or as a per-call keyword.

| Input                  | [`IEEE`](@ref) | [`NanOnlyAllOnes`](@ref) | [`FiniteOnly`](@ref) |
| ---------------------- | -------------- | ------------------------ | -------------------- |
| `isnan(x)`             | NaN            | NaN                      | Error                |
| `abs(x) > floatmax(T)` | ±floatmax      | ±floatmax                | ±floatmax            |

See also [`OVF`](@ref).

# Examples
```jldoctest
julia> @microfloat SaturatingFloat8 exponent=4 significand=3 overflow=Microfloats.SAT

julia> SaturatingFloat8(10000)
SaturatingFloat8(240.0)

julia> Float8_E4M3(10000; overflow=Microfloats.SAT)
Float8_E4M3(240.0)
```
"""
const SAT = Saturating()

"""
    overflow_policy(::Type{<:Microfloat}) -> OverflowPolicy

Return the overflow policy *instance* registered by
[`@microfloat`](@ref) — typically [`OVF`](@ref) or [`SAT`](@ref). Sets
the default for `overflow=...` at every conversion call site for this
type; override per call with the `overflow` keyword.

# Examples
```jldoctest
julia> Microfloats.overflow_policy(Float8_E5M2)
Microfloats.Overflowing()

julia> Microfloats.overflow_policy(Float4_E2M1FN)
Microfloats.Saturating()
```
"""
overflow_policy(::Type{T}) where T<:Microfloat =
    error("$T must define `Microfloats.overflow_policy(::Type{$T})`")

# ───────────────────────── error hooks ──────────────────────────

# Every error path in the conversion kernels routes through one of these
# `@noinline` hooks so device backends (e.g. CUDACoreExt) can override just
# the hooks — via `@device_override` — and run the *same* numeric kernels on
# device, instead of maintaining a duplicated device-safe copy of the whole
# conversion body.
@noinline throw_negative_unsigned(::Type{T}, x) where T =
    throw(DomainError(x, "negative input to unsigned $T"))
@noinline throw_negate_unsigned(::Type{T}, x) where T =
    throw(DomainError(x, "cannot negate unsigned $T"))
@noinline throw_no_nan(::Type{T}, x) where T =
    throw(DomainError(x, "$T has no NaN"))
@noinline throw_no_overflow_sentinel(::Type{T}, x) where T =
    throw(DomainError(x, "$T has no overflow sentinel; use overflow=SAT"))
@noinline throw_unsupported_rounding(::Type{T}, mode) where T =
    throw(ArgumentError("$T does not support rounding mode $mode"))

# ───────────────────────── rounding shifts ──────────────────────────

function rshift_round_to_even(x::T, n::Int) where T<:Unsigned
    n <= 0 && return x >> n
    n > 8 * sizeof(T) && return zero(T)
    mask = (T(1) << n) - T(1)
    half = T(1) << (n - 1)
    lower = x & mask
    up = (lower > half) | ((lower == half) & (((x >> n) & T(1)) == T(1)))
    (x >> n) + (up ? T(1) : T(0))
end

function rshift_round_ties_away(x::T, n::Int) where T<:Unsigned
    n <= 0 && return x >> n
    n > 8 * sizeof(T) && return zero(T)
    mask = (T(1) << n) - T(1)
    half = T(1) << (n - 1)
    lower = x & mask
    up = lower >= half
    (x >> n) + (up ? T(1) : T(0))
end

rshift_truncate(x::T, n::Int) where T<:Unsigned = x >> n

function rshift_round_up_magnitude(x::T, n::Int) where T<:Unsigned
    n <= 0 && return x >> n
    mask = (T(1) << n) - T(1)
    has_low_bits = (x & mask) != T(0)
    (x >> n) + (has_low_bits ? T(1) : T(0))
end

is_outside_floatmax(x::Float32, ::Type{T}) where T<:Microfloat =
    reinterpret(Unsigned, abs(x)) > reinterpret(Unsigned, Float32(floatmax(T)))
clamp_floatmax(x::T) where T<:Microfloat = signbit(x) ? -floatmax(T) : floatmax(T)
clamp_inf(x::T) where T<:Microfloat = signbit(x) ? -inf(T) : inf(T)

@inline mode_overflows_to_inf(::RoundingMode{:Nearest},          ::Bool) = true
@inline mode_overflows_to_inf(::RoundingMode{:NearestTiesAway},  ::Bool) = true
@inline mode_overflows_to_inf(::RoundingMode{:FromZero},         ::Bool) = true
@inline mode_overflows_to_inf(::RoundingMode{:ToZero},           ::Bool) = false
@inline mode_overflows_to_inf(::RoundingMode{:Up},   signbit::Bool) = !signbit
@inline mode_overflows_to_inf(::RoundingMode{:Down}, signbit::Bool) = signbit

function apply_overflow_policy(x::T, xf::Float32, mode::RoundingMode, ::Overflowing) where T<:Microfloat
    if isnan(xf)
        return hasnan(T) ? nan(T) : throw_no_nan(T, xf)
    elseif isinf(xf) || is_outside_floatmax(xf, T)
        if mode_overflows_to_inf(mode, signbit(xf))
            return hasinf(T) ? clamp_inf(x) :
                   hasnan(T) ? nan(T) :
                   throw_no_overflow_sentinel(T, xf)
        else
            return clamp_floatmax(x)
        end
    else
        return x
    end
end

function apply_overflow_policy(x::T, xf::Float32, ::RoundingMode, ::Saturating) where T<:Microfloat
    if isnan(xf)
        return hasnan(T) ? nan(T) : throw_no_nan(T, xf)
    elseif isinf(xf) || is_outside_floatmax(xf, T)
        return clamp_floatmax(x)
    else
        return x
    end
end

# All rounding modes share this body; the `rshift` helper varies.
function _round_to_microfloat(::Type{T}, x::Float32, rshift::F,
                              mode::RoundingMode, policy::OverflowPolicy
                              ) where {T<:Microfloat, F}
    if sign_bits(T) == 0 && signbit(x)
        throw_negative_unsigned(T, x)
    end
    # A signed zero keeps its sign; unsigned formats have an empty sign mask.
    iszero(x) && return reinterpret(T, signbit(x) ? sign_mask(T) : 0x00)

    f32_raw  = reinterpret(UInt32, x)
    f32_exp  = Int((f32_raw >> 23) & UInt32(0x000000ff))
    f32_frac = f32_raw & UInt32(0x007fffff)

    # A subnormal Float32 is normalized first, so `sig24` always carries its
    # leading one in bit 23. Only formats that reach below Float32's normal
    # range (E8M0's 2^-127) can tell the difference.
    nlz = f32_exp == 0 ? leading_zeros(f32_frac) - 8 : 0
    sig24 = f32_exp == 0 ? f32_frac << nlz : (UInt32(0x00800000) | f32_frac)
    true_exp = f32_exp == 0 ? -126 - nlz : (f32_exp - 127)
    t_exp = true_exp + exponent_bias(T)

    if significand_bits(T) == 0 && t_exp < 0
        # Without significand bits there are no subnormals and no zero: the
        # all-zero exponent is the smallest value, and everything below it
        # rounds or saturates to it.
        t_raw = 0x00
    elseif t_exp <= 0 && significand_bits(T) > 0
        # Subnormal path in target format
        shift = t_exp + significand_bits(T) - 24
        sub_q = rshift(sig24, -shift)
        max_frac = UInt32((1 << significand_bits(T)) - 1)
        if sub_q == 0
            t_raw = 0x00
        elseif sub_q == (UInt32(1) << significand_bits(T))
            t_raw = UInt8(1) << significand_bits(T)
        else
            sub_q = min(sub_q, max_frac)
            t_raw = UInt8(sub_q & max_frac)
        end
    else
        # Normal path in target format
        shift = 23 - significand_bits(T)
        total = rshift(sig24, shift)
        if total == 0
            t_raw = 0x00
        else
            t_exp_rounded = t_exp + Int(total >> (significand_bits(T) + 1))
            max_exp = (1 << exponent_bits(T)) - 1
            if t_exp_rounded > max_exp
                t_exp_rounded = max_exp
                if !hasinf(T)
                    total = (UInt32(1) << significand_bits(T)) | UInt32((1 << significand_bits(T)) - 1)
                end
            end
            frac_field = UInt8(total) & UInt8((1 << significand_bits(T)) - 1)
            t_raw = (UInt8(t_exp_rounded) << significand_bits(T)) | frac_field
        end
    end

    t_raw |= (((f32_raw >> 31) % UInt8) << (exponent_bits(T) + significand_bits(T))) & sign_mask(T)

    return apply_overflow_policy(reinterpret(T, t_raw), x, mode, policy)
end

# ───────────────────────── branch-free narrowing ──────────────────────────

# Whether a directed mode rounds the magnitude up, for a value of sign `neg`.
@inline rounds_up(::RoundingMode{:ToZero},   ::Bool) = false
@inline rounds_up(::RoundingMode{:FromZero}, ::Bool) = true
@inline rounds_up(::RoundingMode{:Up},   neg::Bool) = !neg
@inline rounds_up(::RoundingMode{:Down}, neg::Bool) = neg

# `a >> s` rounded per mode. On the bits of a positive Float32 this rounds
# its significand, carrying into the exponent.
@inline round_shift(a::UInt32, s::Int, ::RoundingMode{:Nearest}, ::Bool) =
    (a + ((UInt32(1) << (s - 1)) - 0x1) + ((a >> s) & 0x1)) >> s
@inline round_shift(a::UInt32, s::Int, ::RoundingMode{:NearestTiesAway}, ::Bool) =
    (a + (UInt32(1) << (s - 1))) >> s
@inline round_shift(a::UInt32, s::Int, mode::RoundingMode, neg::Bool) =
    (a + ifelse(rounds_up(mode, neg), (UInt32(1) << s) - 0x1, UInt32(0))) >> s

# `v ≥ 0`, below 2^8, rounded per mode to an integer. Nearest adds 2^23, which
# leaves no fraction bits, so the FPU's own ties-to-even rounding applies.
@inline round_int(v::Float32, ::RoundingMode{:Nearest}, ::Bool) =
    reinterpret(UInt32, v + Float32(0x1p23)) - 0x4b000000
@inline round_int(v::Float32, ::RoundingMode{:NearestTiesAway}, ::Bool) =
    unsafe_trunc(UInt32, round(v, RoundNearestTiesAway))
@inline round_int(v::Float32, mode::RoundingMode, neg::Bool) =
    unsafe_trunc(UInt32, ifelse(rounds_up(mode, neg), ceil(v), v))

# Bits of the Float32 equal to `2^e`, for e ≥ -149.
@inline pow2_bits(e::Int) = e >= -126 ? UInt32(e + 127) << 23 : UInt32(1) << (e + 149)

const TwiddledModes = Union{RoundingMode{:Nearest}, RoundingMode{:NearestTiesAway},
                            RoundingMode{:ToZero}, RoundingMode{:FromZero},
                            RoundingMode{:Up}, RoundingMode{:Down}}

"""
    cvt_twiddle(::Type{T}, x::Float32, mode::RoundingMode, policy::OverflowPolicy) -> T

Branch-free narrowing from `Float32`, the default [`cvt`](@ref) method for
`Float32` sources (and so for every source that converts to `Float32`
first). Results, and errors, are identical to [`cvt_generic`](@ref), which
it calls for modes it does not cover.

Target normals round the `Float32` bits directly: an integer add rounds the
significand at the target's last significand bit, carrying into the
exponent, and a subtraction rebiases. Target subnormals round `|x|` scaled
to units of the smallest subnormal, which the FPU does exactly. Everything
else is a select, so the only branches are the error checks, which the
compiler removes for formats that cannot fail (e.g. signed formats with NaN).
"""
@inline function cvt_twiddle(::Type{T}, x::Float32, mode::TwiddledModes,
                             policy::OverflowPolicy) where T<:Microfloat
    twiddle_fails(T, x, mode, policy) && throw_twiddle_error(T, x, mode, policy)
    return twiddle_bits(T, x, mode, policy)
end

# The magnitude of floatmax(T) and whether |x| exceeds it, which includes ±Inf
# and NaN. The Float32 bits of floatmax(T) fold to a constant.
@inline function exceeds_floatmax(::Type{T}, x::Float32) where T<:Microfloat
    max_mag = reinterpret(UInt8, floatmax(T)) & ~sign_mask(T)
    return max_mag, reinterpret(UInt32, abs(x)) > reinterpret(UInt32, cvt_generic(Float32, floatmax(T)))
end

@inline overflows_to_sentinel(mode::RoundingMode, policy::OverflowPolicy, neg::Bool) =
    policy isa Overflowing && mode_overflows_to_inf(mode, neg)

# Whether the generic path would throw for `x`, as a branch-free Bool. Folds
# to `false` for formats that cannot fail (signed, with NaN), which leaves
# `cvt_twiddle` without branches.
@inline function twiddle_fails(::Type{T}, x::Float32, mode::RoundingMode,
                               policy::OverflowPolicy) where T<:Microfloat
    _, over = exceeds_floatmax(T, x)
    return (sign_bits(T) == 0 && signbit(x)) | (!hasnan(T) && isnan(x)) |
           (!hasinf(T) && !hasnan(T) && over && overflows_to_sentinel(mode, policy, signbit(x)))
end

# Raises the error the generic path raises for `x`, in the same order.
@noinline function throw_twiddle_error(::Type{T}, x::Float32, mode::RoundingMode,
                                       policy::OverflowPolicy) where T<:Microfloat
    sign_bits(T) == 0 && signbit(x) && throw_negative_unsigned(T, x)
    !hasnan(T) && isnan(x) && throw_no_nan(T, x)
    throw_no_overflow_sentinel(T, x)
end

# The result bits of `cvt_twiddle` for an `x` that does not fail.
@inline function twiddle_bits(::Type{T}, x::Float32, mode::TwiddledModes,
                              policy::OverflowPolicy) where T<:Microfloat
    M, bias = significand_bits(T), exponent_bias(T)
    a = reinterpret(UInt32, x) & 0x7fffffff
    neg = signbit(x)
    isnan_x = a > 0x7f800000
    max_mag, over = exceeds_floatmax(T, x)
    sentinel = overflows_to_sentinel(mode, policy, neg)

    # Target normals: round at the target's last significand bit, rebias.
    rebias = UInt32(127 - bias) << M
    t = if M > 0
        normal = round_shift(a, 23 - M, mode, neg) - rebias
        # Below floatmin(T), count units of the smallest subnormal, 2^(1-bias-M).
        scale = reinterpret(Float32, pow2_bits(M + bias - 1))
        sub = round_int(abs(x) * scale, mode, neg)
        ifelse(a < pow2_bits(1 - bias), sub, normal)
    else
        # Without significand bits there are no subnormals and no zero: the
        # all-zero encoding 2^-bias is the smallest value, and everything
        # below it maps to it. It may lie below Float32's normals (E8M0's
        # 2^-127), so Float32 subnormals are scaled to normals first.
        # The generic path's ties-to-even reads the implicit leading one as
        # the last significand bit here, so its ties round away.
        f32_sub = a < 0x00800000
        a_normal = ifelse(f32_sub, reinterpret(UInt32, abs(x) * Float32(0x1p24)), a)
        m = mode isa RoundingMode{:Nearest} ? RoundNearestTiesAway : mode
        normal = round_shift(a_normal, 23, m, neg) - ifelse(f32_sub, rebias + 0x18, rebias)
        ifelse(a < pow2_bits(-bias), 0x00000000, normal)
    end

    # Overflow, then the sign; NaN results carry no sign, like `nan(T)`.
    ovf_mag = if !(policy isa Overflowing)
        max_mag
    elseif hasinf(T)
        ifelse(sentinel, reinterpret(UInt8, inf(T)), max_mag)
    elseif hasnan(T)
        ifelse(sentinel, reinterpret(UInt8, nan(T)), max_mag)
    else
        max_mag
    end
    r = ifelse(over, ovf_mag, t % UInt8)
    sign_bits(T) == 1 && (r |= ifelse(neg, sign_mask(T), 0x00))
    if hasnan(T)
        nan_result = isnan_x | (over & sentinel & !hasinf(T))
        r = ifelse(nan_result, reinterpret(UInt8, nan(T)), r)
    end
    return reinterpret(T, r)
end

cvt_twiddle(::Type{T}, x::Float32, mode::RoundingMode, policy::OverflowPolicy) where T<:Microfloat =
    cvt_generic(T, x, mode, policy)

# ───────────────────────── conversion funnel ──────────────────────────

"""
    cvt(::Type{T}, x, mode::RoundingMode, policy::OverflowPolicy) -> T

Central conversion funnel. Every scalar conversion into a
[`Microfloat`](@ref) — constructors, `convert`, broadcasts, and the packed
vector paths — reduces to a call of this function, with the rounding mode
and overflow policy as positional, dispatchable arguments.

`cvt` is the extension surface for optimized conversions. To specialize,
add a method on any subset of `(T, typeof(x), mode, policy)`:

- **Bit-twiddling / table specializations** add ordinary methods, e.g.
  `Microfloats.cvt(::Type{Float8_E4M3}, x::Float4_E2M1FN, ::RoundingMode,
  ::OverflowPolicy)`. See [`@cvt_table`](@ref) for a generated lookup-table
  shortcut.
- **Device backends** (package extensions) use overlay method tables (e.g.
  `CUDACore.@device_override`) on exactly the `(T, source, mode, policy)`
  signatures the hardware supports natively; every other combination falls
  through to the portable methods below.

The always-correct reference path is [`cvt_generic`](@ref); specialized
methods that need a partial fallback should call it (not `cvt`, which on
overlay method tables would recurse into the override itself).
"""
@inline cvt(::Type{T}, x::Real, mode::RoundingMode, policy::OverflowPolicy) where T<:Microfloat =
    cvt(T, Float32(x), mode, policy)
@inline cvt(::Type{T}, x::Float32, mode::RoundingMode, policy::OverflowPolicy) where T<:Microfloat =
    cvt_twiddle(T, x, mode, policy)
@inline cvt(::Type{T}, x::Microfloat, mode::RoundingMode, policy::OverflowPolicy) where T<:Microfloat =
    cvt_generic(T, Float32(x), mode, policy)

"""
    cvt_generic(::Type{T}, x::Float32, mode::RoundingMode, policy::OverflowPolicy) -> T

The generic reference implementation behind [`cvt`](@ref): bit-level
rounding from `Float32` into any `Microfloat` layout, for every supported
rounding mode and overflow policy. Specialized `cvt` methods (and device
overrides) call this directly when their fast path does not apply.
"""
@inline cvt_generic(::Type{T}, x::Float32, mode::RoundingMode{:Nearest}, policy::OverflowPolicy) where T<:Microfloat =
    _round_to_microfloat(T, x, rshift_round_to_even, mode, policy)
@inline cvt_generic(::Type{T}, x::Float32, mode::RoundingMode{:NearestTiesAway}, policy::OverflowPolicy) where T<:Microfloat =
    _round_to_microfloat(T, x, rshift_round_ties_away, mode, policy)
@inline cvt_generic(::Type{T}, x::Float32, mode::RoundingMode{:ToZero}, policy::OverflowPolicy) where T<:Microfloat =
    _round_to_microfloat(T, x, rshift_truncate, mode, policy)
@inline cvt_generic(::Type{T}, x::Float32, mode::RoundingMode{:FromZero}, policy::OverflowPolicy) where T<:Microfloat =
    _round_to_microfloat(T, x, rshift_round_up_magnitude, mode, policy)

# RoundUp/RoundDown are sign-dependent: "toward +∞" rounds the magnitude up
# for positive inputs but truncates the magnitude for negative inputs (which
# moves the value toward zero, i.e., closer to +∞). RoundDown is the mirror.
@inline cvt_generic(::Type{T}, x::Float32, mode::RoundingMode{:Up}, policy::OverflowPolicy) where T<:Microfloat =
    signbit(x) ? _round_to_microfloat(T, x, rshift_truncate,           mode, policy) :
                 _round_to_microfloat(T, x, rshift_round_up_magnitude, mode, policy)
@inline cvt_generic(::Type{T}, x::Float32, mode::RoundingMode{:Down}, policy::OverflowPolicy) where T<:Microfloat =
    signbit(x) ? _round_to_microfloat(T, x, rshift_round_up_magnitude, mode, policy) :
                 _round_to_microfloat(T, x, rshift_truncate,           mode, policy)

cvt_generic(::Type{T}, x::Float32, mode::RoundingMode, ::OverflowPolicy) where T<:Microfloat =
    throw_unsupported_rounding(T, mode)

# ───────────────────────── constructors ──────────────────────────

# Constructors are thin sugar over `cvt`: they only resolve defaults
# (RoundNearest, the type's registered overflow policy) and are never
# specialized or device-overridden themselves.
#
# `Real` (not `Number`) avoids colliding with Base's
# `(::Type{T})(::Real, ::RoundingMode) where T<:AbstractFloat`.
(::Type{T})(x::Real;
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    cvt(T, x, RoundNearest, overflow)
(::Type{T})(x::Real, mode::RoundingMode;
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    cvt(T, x, mode, overflow)
# `Rational` sources: otherwise ambiguous with Base's
# `(::Type{T})(::Rational) where T<:AbstractFloat`.
(::Type{T})(x::Rational{S};
            overflow::OverflowPolicy = overflow_policy(T)) where {S,T<:Microfloat} =
    cvt(T, x, RoundNearest, overflow)

# Returns the BFloat16 encoding as raw UInt16 bits. Internal plumbing stays
# in bits because on Julia >= 1.12 a BFloat16 *value* crossing a function-call
# boundary has subnormals flushed to zero on CPUs with native BF16
# instructions (e.g. Zen 4), which corrupts E8M0's 2^-127.
function _to_bfloat16_bits(x::T) where {T<:Microfloat}
    t_raw = reinterpret(UInt8, x)

    t_sign = (sign_bits(T) == 1) && (t_raw & (UInt8(1) << (exponent_bits(T) + significand_bits(T))) != 0)
    t_exponent_field = Int((t_raw >> significand_bits(T)) & UInt8((1 << exponent_bits(T)) - 1))
    t_fraction_field = UInt16(t_raw & UInt8((1 << significand_bits(T)) - 1))

    bf16_sign_bit = UInt16(t_sign ? 1 : 0) << 15

    if isinf(x)
        return bf16_sign_bit | 0x7f80
    elseif isnan(x)
        return bf16_sign_bit | 0x7fc0
    elseif iszero(x)
        return bf16_sign_bit
    end

    M = significand_bits(T)
    bias = exponent_bias(T)

    if t_exponent_field == 0 && M > 0 # Subnormal
        nlz = leading_zeros(t_fraction_field) + M - 16
        t_significand_total = UInt16(t_fraction_field) << (nlz + 1)
        t_true_exponent = -nlz - bias
    else # Normal
        t_significand_total = (UInt16(1) << M) + t_fraction_field
        t_true_exponent = t_exponent_field - bias
    end

    bf16_exponent_field = t_true_exponent + 127
    bf16_significand_total = if M >= 7
        rshift_round_to_even(t_significand_total, M - 7)
    else
        t_significand_total << (7 - M)
    end
    if bf16_significand_total == 0x0100
        bf16_significand_total = 0x0080
        bf16_exponent_field += 1
    end
    if bf16_exponent_field >= 0xff
        return bf16_sign_bit | 0x7f80
    elseif bf16_exponent_field <= 0
        shift_to_bf16_sub = t_true_exponent + 133 - M
        sub_q = shift_to_bf16_sub >= 0 ? (t_significand_total << shift_to_bf16_sub) : rshift_round_to_even(t_significand_total, -shift_to_bf16_sub)
        if sub_q == 0
            return bf16_sign_bit
        elseif sub_q >= 0x80
            return bf16_sign_bit | UInt16(0x0080)
        else
            return bf16_sign_bit | UInt16(sub_q & 0x7f)
        end
    else
        return bf16_sign_bit | (UInt16(bf16_exponent_field & 0xff) << 7) | UInt16((bf16_significand_total - 0x80) & 0x7f)
    end
end

# `@microfloat` adds a new method to `to_bfloat16_bits`
function to_bfloat16_bits end

to_bfloat16(x::Microfloat) = reinterpret(BFloat16, to_bfloat16_bits(x))

function _format_sci(f64::Float64, n::Int)
    ax = abs(f64)
    e = floor(Int, log10(ax))
    k = n - 1 - e
    scaled = k >= 0 ? ax * exp10(k) : ax / exp10(-k)
    m = round(Int, scaled)
    if m >= 10^n
        m ÷= 10
        e += 1
    end
    digits = lpad(string(m), n, '0')
    mantissa = n == 1 ? digits * ".0" : digits[1:1] * "." * digits[2:n]
    return (signbit(f64) ? "-" : "") * mantissa * "e" * string(e)
end

# Try increasing precision until the rounded decimal round-trips through `T`.
# 4 sig digits suffices on precision grounds, but near floatmax under an OVF
# policy the rounded value can land in the NaN sentinel for several ndig in a
# row — empirically up to 6 (e.g. `_E1M7_NaN(0xfe) == 3.96875`). 9 is generous
# headroom given the 8-bit-total constraint. When Ryu's shortest form blows up
# on values not exactly representable in Float64
# (e.g. "2.9999999999999998e-40"), detect via length and reformat.
function _shortest_decimal_string(x::T) where T<:Microfloat
    isnan(x) && return "NaN"
    isinf(x) && return signbit(x) ? "-Inf" : "Inf"
    iszero(x) && return signbit(x) ? "-0.0" : "0.0"
    f64 = Float64(x)
    for ndig in 1:9
        rounded = round(f64, sigdigits=ndig)
        T(rounded) === x || continue
        s = string(rounded)
        length(s) <= ndig + 8 && return s
        return _format_sci(f64, ndig)
    end
    error("unreachable: no round-tripping decimal in 1:9 sig digits for $T")
end

# `@microfloat` adds a new method to `decimal_string`
function decimal_string end

# ───────────────────────── widening funnel ──────────────────────────

"""
    WideFloat

The floating-point destinations of the widening funnel: `Float16`,
`BFloat16`, `Float32` and `Float64`.
"""
const WideFloat = Union{Float16,BFloat16,Float32,Float64}

"""
    cvt(::Type{F}, x::Microfloat) -> F

Widening half of the conversion funnel: every conversion *out of* a
[`Microfloat`](@ref) into `Float16`, `BFloat16`, `Float32` or `Float64`
reduces to a call of this method, the same way conversions into a
`Microfloat` reduce to the four-argument form.

Every `Microfloat` is exactly representable in `BFloat16` (at most 7
significand bits and 8 exponent bits), and so in `Float32` and `Float64`:
widening never rounds, which is why this form takes no rounding mode or
overflow policy. `Float16` has a narrower exponent range than some formats
(`Float8_E8M0FNU`); it receives the exact `Float32` value rounded by
`Float16(::Float32)`.

`@microfloat` registers a method per type that computes it with a
branch-free bit-twiddle over the destination's bits where one is cheap (see
[`max_twiddle_cost`](@ref)), and with the generic lookup otherwise; `Float64`
extends the `Float32` result. Like the narrowing form this is the extension
surface for device backends, which override it for the destinations their
hardware widens to natively.
"""
@inline cvt(::Type{F}, x::Microfloat) where F<:WideFloat = cvt_generic(F, x)

# Reference widening path, the counterpart of the narrowing `cvt_generic`:
# a lookup of the BFloat16 encoding generated per type by `@microfloat`.
@inline cvt_generic(::Type{Float32}, x::Microfloat) =
    # Shift the BFloat16 *encoding* into a Float32 instead of materializing a
    # BFloat16: on Julia >= 1.12, a BFloat16 value crossing a function-call
    # boundary has subnormals flushed to zero on CPUs with native BF16
    # instructions (e.g. Zen 4), which corrupts E8M0's 2^-127.
    reinterpret(Float32, UInt32(to_bfloat16_bits(x)) << 16)
@inline cvt_generic(::Type{BFloat16}, x::Microfloat) = to_bfloat16(x)
@inline cvt_generic(::Type{F}, x::Microfloat) where F<:Union{Float16,Float64} =
    F(cvt_generic(Float32, x))

# Constructors are sugar over the funnel, as on the narrowing side. Base then
# supplies `T(::Float32)` for the remaining numeric types (Int*, BigFloat).
# One method per destination: a `Union`-bounded type variable would be
# ambiguous with constructors such as `BFloat16(::AbstractFloat)`.
for F in (Float16, BFloat16, Float32, Float64)
    @eval (::Type{$F})(x::Microfloat) = cvt($F, x)
end
(::Type{T})(x::Microfloat) where T<:Number = T(cvt(Float32, x))
# Base constructors that are more specific in the destination but less
# specific in the source than the catch-all above.
Base.Complex{T}(x::Microfloat) where T<:Real = Complex{T}(T(x), zero(T))
Base.Complex(x::Microfloat) = Complex(x, zero(x))
Base.Rational{T}(x::Microfloat) where T<:Integer = Rational{T}(cvt(Float32, x))
Base.Rational{BigInt}(x::Microfloat) = Rational{BigInt}(cvt(Float32, x))

# Microfloat → Microfloat: disambiguates the methods above; the default
# `cvt` route goes through Float32 (matching the Real-input path and avoiding
# the BFloat16 intermediate's narrower exponent dynamic range) unless a
# specialized `cvt` method — e.g. one registered by `@cvt_table` — applies.
(::Type{T})(x::Microfloat;
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    cvt(T, x, RoundNearest, overflow)

# ───────────────────────── @cvt_table ──────────────────────────

# Raw result bits of `cvt_generic` for every `S` bit pattern, so anything
# generated from it is correct by construction. `nothing` if any pattern
# throws (e.g. negative values into an unsigned target): the combination then
# keeps the runtime generic path, and its errors.
function cvt_generic_table(::Type{T}, ::Type{S}, mode::RoundingMode, policy::OverflowPolicy
                           ) where {T<:Microfloat, S<:Microfloat}
    table = UInt8[]
    for raw in 0:(1 << bitwidth(S)) - 1
        y = try
            cvt_generic(T, Float32(reinterpret(S, raw % UInt8)), mode, policy)
        catch
            return nothing
        end
        push!(table, reinterpret(UInt8, y))
    end
    return table
end

# Conversion tables are mostly piecewise linear in their index. An exact
# widening is one linear piece over the source normals (a fixed exponent
# offset added to the shifted bits, carries between exponent and significand
# included), plus a piece per binade of source subnormals that the target
# normalizes; saturated and NaN results are constant runs. A piece
# `(lo, a, c)` maps each index `i` from `lo` up to the next piece to
# `a * i + c` in the table's unsigned arithmetic, with `a` zero or a power of
# two, so it costs a shift and an add. Returns the fewest pieces covering
# `table`, whose first index is `first`.
function linear_pieces(table::AbstractVector{U}, first::Int = 0) where U<:Unsigned
    n = length(table)
    # The piece starting at index `i` (1-based) takes its slope from its
    # first two entries and extends as far as they predict: `stop[i]`.
    stop = Vector{Int}(undef, n)
    piece = Vector{Tuple{Int,U,U}}(undef, n)
    for i in 1:n
        k = U(first + i - 1)
        a = i < n ? table[i + 1] - table[i] : zero(U)
        iszero(a) || ispow2(a) || (a = zero(U))
        c = table[i] - a * k
        j = i
        while j < n && a * U(first + j) + c == table[j + 1]
            j += 1
        end
        stop[i], piece[i] = j, (first + i - 1, a, c)
    end
    # Fewest pieces for each prefix, then walk back from the full table.
    count = fill(typemax(Int), n + 1)
    from = zeros(Int, n + 1)
    count[1] = 0
    for i in 1:n, j in i:stop[i]
        if count[i] + 1 < count[j + 1]
            count[j + 1], from[j + 1] = count[i] + 1, i
        end
    end
    pieces = Tuple{Int,U,U}[]
    j = n + 1
    while j > 1
        pushfirst!(pieces, piece[from[j]])
        j = from[j]
    end
    return pieces
end

function piece_expr(i::Symbol, (_, a, c)::Tuple{Int,U,U}) where U
    iszero(a) && return c
    ai = isone(a) ? i : :($i << $(trailing_zeros(a)))
    return iszero(c) ? ai : :($ai + $c)
end

"""
    max_twiddle_cost() -> Int

Largest cost of a bit-twiddle that generated conversion methods (see
[`@cvt_table`](@ref)) prefer over a table lookup: its pieces, plus one if it
moves a sign bit. Each piece is a shift, an add, a compare and a select. On
CPUs the twiddle vectorizes where a lookup is a gather, and wins by a wide
margin up to about twice this many pieces. Device backends override it: on
GPUs a load from a small constant table hits the read-only cache and beats
all but the shortest twiddles.
"""
max_twiddle_cost() = 8

# A bit-twiddle computing a conversion table of `U` results from the source
# bits: the index `i` is the source bits under `mask`. Indices below `head`
# take a head expression (see `widen_subnormal`), the rest `pieces`. When
# `shift` is not `nothing`, the index is the magnitude and the source sign
# bit moves `shift` bits up.
struct Twiddle{U<:Unsigned}
    pieces::Vector{Tuple{Int,U,U}}
    mask::UInt8
    shift::Union{Int,Nothing}
    head::Int
end

twiddle_cost(tw::Twiddle) = length(tw.pieces) + (tw.head > 0) + (tw.shift !== nothing)

# The cheapest twiddle for the table of `S` conversions into a format whose
# sign bit is `sign` (`nothing` if unsigned). The whole table always fits;
# when the sign moves across unchanged, the magnitude half alone usually fits
# in fewer pieces. `head` is `(n, f)` if `f(i)` computes the first `n`
# magnitudes by other means; it is used where it matches the table.
function twiddle_fit(table::Vector{U}, ::Type{S}, sign::Union{Int,Nothing};
                     head = nothing) where {U<:Unsigned, S<:Microfloat}
    n = length(table)
    candidates = [Twiddle{U}(linear_pieces(table), UInt8(n - 1), nothing, 0)]
    magnitude(h, shift) = if head !== nothing && head[1] < h &&
                             all(head[2](i) == table[i + 1] for i in 0:head[1] - 1)
        Twiddle{U}(linear_pieces(table[head[1] + 1:h], head[1]), UInt8(h - 1), shift, head[1])
    else
        Twiddle{U}(linear_pieces(table[1:h]), UInt8(h - 1), shift, 0)
    end
    if sign_bits(S) == 0
        push!(candidates, magnitude(n, nothing))
    elseif sign !== nothing
        h = n >> 1
        if all(table[h + m + 1] == table[m + 1] | (one(U) << sign) for m in 0:h - 1)
            push!(candidates, magnitude(h, sign - (bitwidth(S) - 1)))
        end
    end
    return argmin(twiddle_cost, candidates)
end

# Branch-free evaluation of a twiddle on `x::S`: every piece is computed and
# the last one starting at or below the index is selected, which vectorizes
# where a table lookup would be a gather.
function twiddle_expr(tw::Twiddle{U}, ::Type{S}, head_expr = nothing) where {U, S}
    ex = tw.head > 0 ? head_expr : piece_expr(:i, tw.pieces[1])
    for p in (tw.head > 0 ? tw.pieces : tw.pieces[2:end])
        ex = :(ifelse(i >= $(U(p[1])), $(piece_expr(:i, p)), $ex))
    end
    body = :(raw = reinterpret(UInt8, x); i = $U(raw & $(tw.mask)); t = $ex)
    if tw.shift !== nothing
        s = :($U(raw & $(sign_mask(S))))
        push!(body.args, :(t |= $(tw.shift >= 0 ? :($s << $(tw.shift)) : :($s >> $(-tw.shift)))))
    end
    push!(body.args, :t)
    return body
end

# Where both a twiddle and a lookup are candidates, the choice is
# `max_twiddle_cost()`, which folds when the method is compiled, so device
# overlays can pick differently than the host. Beyond twice the host default
# a lookup wins everywhere.
function twiddle_or(lookup, tw::Twiddle, twiddle)
    cost = twiddle_cost(tw)
    cost > 2 * max_twiddle_cost() && return lookup
    return :($cost <= $max_twiddle_cost() ? $twiddle : $lookup)
end

# The cheapest correct code for one (T, S, mode, policy) combination: a
# bit-twiddle when the generic results are piecewise linear in few pieces, a
# `2^bitwidth(S)`-entry lookup otherwise.
function cvt_expr(::Type{T}, ::Type{S}, ::Type{M}, ::Type{P}
                  ) where {T<:Microfloat, S<:Microfloat, M<:RoundingMode, P<:OverflowPolicy}
    table = cvt_generic_table(T, S, M.instance, P.instance)
    table === nothing && return :($cvt_generic($T, Float32(x), mode, policy))
    lookup = :(reinterpret($T, $(Tuple(table))[Int(reinterpret(UInt8, x) & $(UInt8(length(table) - 1))) + 1]))
    tw = twiddle_fit(table, S, sign_bits(T) == 1 ? bitwidth(T) - 1 : nothing)
    return twiddle_or(lookup, tw, :(reinterpret($T, $(twiddle_expr(tw, S)))))
end

# ───────────────────────── generated widening ──────────────────────────

wide_bits(::Type{F}) where F<:WideFloat = Base.uinttype(F)
wide_bits(::Type{BFloat16}) = UInt16

# Zero and the subnormals of `S`, widened to the bits of `F` through the
# FPU: under the exponent of 2^(1-bias), the significand bits `i` read as
# 2^(1-bias) * (1 + i/2^M), and subtracting 2^(1-bias) leaves exactly
# i * 2^(1-bias-M), normalized. No operand is subnormal. One head piece in
# place of a linear piece per binade of subnormals.
@inline function widen_subnormal(::Type{F}, ::Type{S}, i) where {F<:Union{Float16,BFloat16,Float32}, S<:Microfloat}
    magic = UInt32(1 - exponent_bias(S) + 127) << 23
    v = reinterpret(Float32, (UInt32(i) << (23 - significand_bits(S))) | magic) - reinterpret(Float32, magic)
    F === Float16 && return reinterpret(UInt16, Float16(v))
    F === BFloat16 && return (reinterpret(UInt32, v) >> 16) % UInt16
    return reinterpret(UInt32, v)
end

# Widening of `S` to `F`, generated per type by `@microfloat`: a bit-twiddle
# over the bits of `F` when the generic results allow a cheap one, else the
# generic lookup. Float64 extends the Float32 result, one instruction, which
# beats a twiddle on 64-bit lanes (half as many per vector).
function widen_expr(::Type{F}, ::Type{S}) where {F<:WideFloat, S<:Microfloat}
    F === Float64 && return :(Float64($cvt(Float32, x)))
    U = wide_bits(F)
    table = [reinterpret(U, cvt_generic(F, reinterpret(S, raw % UInt8))) for raw in 0:(1 << bitwidth(S)) - 1]
    head = significand_bits(S) > 0 ? (1 << significand_bits(S), i -> widen_subnormal(F, S, i)) : nothing
    tw = twiddle_fit(table, S, 8 * sizeof(U) - 1; head)
    twiddle = :(reinterpret($F, $(twiddle_expr(tw, S, :($widen_subnormal($F, $S, i))))))
    return twiddle_or(:($cvt_generic($F, x)), tw, twiddle)
end

"""
    @cvt_table Src => Dst

Register an optimized method on the conversion funnel [`cvt`](@ref) for
converting microfloat `Src` values to microfloat `Dst`.

Expands to a `@generated` method of `Microfloats.cvt`. Once per
`(mode, policy)` combination actually used, it runs every `Src` bit pattern
through [`cvt_generic`](@ref) and compiles the resulting table into the
cheapest equivalent code, so results are identical to the generic path:

- a branch-free bit-twiddle when the table is piecewise linear in a few
  pieces, as for exact widenings like `Float4_E2M1FN => Float8_E4M3FN`
  (each piece is a shift and an add, selected by comparing the source bits);
- otherwise a single `2^bitwidth(Src)`-entry table lookup.

Which counts as "a few" is [`max_twiddle_cost`](@ref), which device
backends lower.

Combinations where the generic path throws (e.g. signed source into unsigned
target) keep the runtime path and its errors.

Invoke *after* both types are defined. Microfloats registers methods for all
pairs of built-in types; user-defined `@microfloat` types can opt in:

```julia
@microfloat MyFloat6 exponent=3 significand=2
Microfloats.@cvt_table MyFloat6 => Float8_E4M3
Microfloats.@cvt_table Float8_E4M3 => MyFloat6
```

To hand-optimize a pair instead, define the `Microfloats.cvt` method for it
directly rather than invoking `@cvt_table` for that pair.
"""
macro cvt_table(pair)
    (pair isa Expr && pair.head === :call && pair.args[1] === :(=>)) ||
        throw(ArgumentError("@cvt_table expects `Src => Dst`, got `$pair`"))
    S, T = pair.args[2], pair.args[3]
    ex = quote
        Base.@generated function $(@__MODULE__).cvt(::Type{$T}, x::$S,
                                                    mode::$RoundingMode, policy::$OverflowPolicy)
            $cvt_expr($T, $S, mode, policy)
        end
        nothing
    end
    return esc(ex)
end
