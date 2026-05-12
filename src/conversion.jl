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
        return hasnan(T) ? nan(T) : throw(DomainError(xf, "$T has no NaN"))
    elseif isinf(xf) || is_outside_floatmax(xf, T)
        if mode_overflows_to_inf(mode, signbit(xf))
            return hasinf(T) ? clamp_inf(x) :
                   hasnan(T) ? nan(T) :
                   throw(DomainError(xf, "$T has no overflow sentinel; use overflow=SAT"))
        else
            return clamp_floatmax(x)
        end
    else
        return x
    end
end

function apply_overflow_policy(x::T, xf::Float32, ::RoundingMode, ::Saturating) where T<:Microfloat
    if isnan(xf)
        return hasnan(T) ? nan(T) : throw(DomainError(xf, "$T has no NaN"))
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
        throw(DomainError(x, "negative input to unsigned $T"))
    end
    iszero(x) && return signbit(x) ? -zero(T) : zero(T)

    f32_raw  = reinterpret(UInt32, x)
    f32_exp  = Int((f32_raw >> 23) & UInt32(0x000000ff))
    f32_frac = f32_raw & UInt32(0x007fffff)

    sig24 = f32_exp == 0 ? f32_frac : (UInt32(0x00800000) | f32_frac)
    true_exp = f32_exp == 0 ? -126 : (f32_exp - 127)
    t_exp = true_exp + exponent_bias(T)

    if t_exp <= 0
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

(::Type{T})(x::Float32, mode::RoundingMode{:Nearest};
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    _round_to_microfloat(T, x, rshift_round_to_even, mode, overflow)
(::Type{T})(x::Float32, mode::RoundingMode{:NearestTiesAway};
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    _round_to_microfloat(T, x, rshift_round_ties_away, mode, overflow)
(::Type{T})(x::Float32, mode::RoundingMode{:ToZero};
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    _round_to_microfloat(T, x, rshift_truncate, mode, overflow)
(::Type{T})(x::Float32, mode::RoundingMode{:FromZero};
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    _round_to_microfloat(T, x, rshift_round_up_magnitude, mode, overflow)

# RoundUp/RoundDown are sign-dependent: "toward +∞" rounds the magnitude up
# for positive inputs but truncates the magnitude for negative inputs (which
# moves the value toward zero, i.e., closer to +∞). RoundDown is the mirror.
(::Type{T})(x::Float32, mode::RoundingMode{:Up};
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    signbit(x) ? _round_to_microfloat(T, x, rshift_truncate,           mode, overflow) :
                 _round_to_microfloat(T, x, rshift_round_up_magnitude, mode, overflow)
(::Type{T})(x::Float32, mode::RoundingMode{:Down};
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    signbit(x) ? _round_to_microfloat(T, x, rshift_round_up_magnitude, mode, overflow) :
                 _round_to_microfloat(T, x, rshift_truncate,           mode, overflow)

# Errors on unsupported modes instead of recursing through the Real-level fallback below.
(::Type{T})(x::Float32, mode::RoundingMode;
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    throw(ArgumentError("$T does not support rounding mode $mode"))

# `Real` (not `Number`) avoids colliding with Base's
# `(::Type{T})(::Real, ::RoundingMode) where T<:AbstractFloat`.
(::Type{T})(x::Real;
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    T(x, RoundNearest; overflow=overflow)
(::Type{T})(x::Real, mode::RoundingMode;
            overflow::OverflowPolicy = overflow_policy(T)) where T<:Microfloat =
    T(Float32(x), mode; overflow=overflow)

function _to_bfloat16(x::T) where {T<:Microfloat}
    t_raw = reinterpret(UInt8, x)

    t_sign = (sign_bits(T) == 1) && (t_raw & (UInt8(1) << (exponent_bits(T) + significand_bits(T))) != 0)
    t_exponent_field = Int((t_raw >> significand_bits(T)) & UInt8((1 << exponent_bits(T)) - 1))
    t_fraction_field = UInt16(t_raw & UInt8((1 << significand_bits(T)) - 1))

    bf16_sign_bit = UInt16(t_sign ? 1 : 0) << 15

    if isinf(x)
        return reinterpret(BFloat16, bf16_sign_bit | 0x7f80)
    elseif isnan(x)
        return reinterpret(BFloat16, bf16_sign_bit | 0x7fc0)
    elseif iszero(x)
        return reinterpret(BFloat16, bf16_sign_bit)
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
        return reinterpret(BFloat16, bf16_sign_bit | 0x7f80)
    elseif bf16_exponent_field <= 0
        shift_to_bf16_sub = t_true_exponent + 133 - M
        sub_q = shift_to_bf16_sub >= 0 ? (t_significand_total << shift_to_bf16_sub) : rshift_round_to_even(t_significand_total, -shift_to_bf16_sub)
        if sub_q == 0
            return reinterpret(BFloat16, bf16_sign_bit)
        elseif sub_q >= 0x80
            return reinterpret(BFloat16, bf16_sign_bit | UInt16(0x0080))
        else
            return reinterpret(BFloat16, bf16_sign_bit | UInt16(sub_q & 0x7f))
        end
    else
        bf16_raw_out = bf16_sign_bit | (UInt16(bf16_exponent_field & 0xff) << 7) | UInt16((bf16_significand_total - 0x80) & 0x7f)
        return reinterpret(BFloat16, bf16_raw_out)
    end
end

# `@microfloat` adds a new method to `to_bfloat16`
function to_bfloat16 end

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

# 4 sig digits cover every Microfloat variant. When the rounded value isn't
# Float64-representable, Ryu's shortest form blows up
# (e.g. "2.9999999999999998e-40"); detect via length and reformat.
function _shortest_decimal_string(x::T) where T<:Microfloat
    isnan(x) && return "NaN"
    isinf(x) && return signbit(x) ? "-Inf" : "Inf"
    iszero(x) && return signbit(x) ? "-0.0" : "0.0"
    f64 = Float64(x)
    for ndig in 1:4
        rounded = round(f64, sigdigits=ndig)
        T(rounded) === x || continue
        s = string(rounded)
        length(s) <= ndig + 8 && return s
        return _format_sci(f64, ndig)
    end
    return string(f64)
end

# `@microfloat` adds a new method to `decimal_string`
function decimal_string end

# Every Microfloat is exactly representable in BFloat16 (M ≤ 7, E ≤ 8), so
# widening through BFloat16 is lossless.
BFloat16(x::T) where T<:Microfloat = to_bfloat16(x)
(::Type{Float32})(x::Microfloat) = Float32(BFloat16(x))
(::Type{Float64})(x::Microfloat) = Float64(BFloat16(x))
(::Type{Float16})(x::Microfloat) = Float16(BFloat16(x))
