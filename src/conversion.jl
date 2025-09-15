abstract type OverflowPolicy end
abstract type OVF <: OverflowPolicy end
abstract type SAT <: OverflowPolicy end

function rshift_round_to_even(x::UInt16, n::Int)
    n <= 0 && return x << (-n)
    lower = x & ((UInt16(1) << n) - UInt16(1))
    half = UInt16(1) << (n - 1)
    up = (lower > half) | ((lower == half) & (((x >> n) & UInt16(1)) == UInt16(1)))
    (x >> n) + (up ? UInt16(1) : UInt16(0))
end

is_outside_floatmax(x::T) where {T<:Microfloat} = reinterpret(Unsigned, abs(x)) > reinterpret(Unsigned, floatmax(T))
clamp_floatmax(x::T) where {T<:Microfloat} = signbit(x) ? -floatmax(T) : floatmax(T)
clamp_inf(x::T) where {T<:Microfloat} = signbit(x) ? -inf(T) : inf(T)

function epilogue(x::T, xb::BFloat16, ::Type{P}) where {T<:Microfloat,P<:OverflowPolicy}
    if P <: SAT
        if isnan(xb)
            return nan(T)
        elseif isinf(xb) || is_outside_floatmax(x)
            return clamp_floatmax(x)
        else
            return x
        end
    elseif P <: OVF
        if isnan(xb)
            return nan(T)
        elseif isinf(xb) || is_outside_floatmax(x)
            return hasinf(T) ? clamp_inf(x) : nan(T)
        else
            return x
        end
    end
end

function (::Type{T})(x::BFloat16, ::Type{P}=OVF) where {T<:Microfloat,P<:OverflowPolicy}
    iszero(x) && return zero(T)

    bf16_exp  = Int((reinterpret(Unsigned, x) >> 7) & 0x00ff)
    bf16_frac = UInt16(reinterpret(Unsigned, x) & 0x007f)

    sig8 = bf16_exp == 0 ? bf16_frac : (0x0080 | bf16_frac)
    true_exp = bf16_exp == 0 ? -126 : (bf16_exp - 127)
    t_exp = true_exp + exponent_bias(T)

    t_raw = 0x00
    if t_exp <= 0
        # Subnormal path in target format
        shift = t_exp + significand_bits(T) - 8
        sub_q = rshift_round_to_even(sig8, -shift)
        max_frac = UInt16((1 << significand_bits(T)) - 1)
        if sub_q == 0
            t_raw = 0x00
        elseif sub_q == (UInt16(1) << significand_bits(T))
            t_raw = UInt8(1) << significand_bits(T)
        else
            sub_q = min(sub_q, max_frac)
            t_raw = UInt8(sub_q & max_frac)
        end
    else
        # Normal path in target format
        shift = 7 - significand_bits(T)
        total = shift >= 0 ? rshift_round_to_even(sig8, shift) : (sig8 << (-shift))
        t_exp_rounded = t_exp + (total >> (significand_bits(T) + 1))
        max_exp = (1 << exponent_bits(T)) - 1
        if t_exp_rounded > max_exp
            if !hasinf(T)
                t_exp_rounded = max_exp
                total = (UInt16(1) << significand_bits(T)) | UInt16((1 << significand_bits(T)) - 1)
            else
                t_exp_rounded = max_exp
            end
        end
        frac_field = UInt8(total) & UInt8((1 << significand_bits(T)) - 1)
        t_raw = (UInt8(t_exp_rounded) << significand_bits(T)) | frac_field
    end

    t_raw |= (reinterpret(Unsigned, x) >> 15 % UInt8) << (exponent_bits(T) + significand_bits(T)) & sign_mask(T)

    return epilogue(reinterpret(T, t_raw), x, P)
end

(::Type{T})(x::Number, args...) where {T<:Microfloat} = T(BFloat16(x), args...)
(::Type{T})(::Type{P}) where {T<:Microfloat,P<:OverflowPolicy} = x -> T(x, P)

function to_bfloat16(x::T) where {T<:Microfloat}
    t_raw = reinterpret(UInt8, x)

    t_sign = (sign_bits(T) == 1) && (t_raw & (UInt8(1) << (exponent_bits(T) + significand_bits(T))) != 0)
    t_exponent_field = Int((t_raw >> significand_bits(T)) & UInt8((1 << exponent_bits(T)) - 1))
    t_fraction_field = UInt16(t_raw & UInt8((1 << significand_bits(T)) - 1))

    bf16_sign_bit = UInt16(t_sign ? 1 : 0) << 15

    # Check for special values first, using trait-aware detection
    if isinf(x)
        return reinterpret(BFloat16, bf16_sign_bit | 0x7f80)
    elseif isnan(x)
        return reinterpret(BFloat16, bf16_sign_bit | 0x7fc0)
    elseif iszero(x)
        return reinterpret(BFloat16, bf16_sign_bit)
    end

    M = significand_bits(T)
    bias = exponent_bias(T)

    if t_exponent_field == 0 # Subnormal
        nlz = M - 1 - floor(Int, log2(t_fraction_field))
        t_significand_total = UInt16(t_fraction_field) << (nlz + 1)
        t_true_exponent = -nlz - bias
    else # Normal
        t_significand_total = (UInt16(1) << M) + t_fraction_field
        t_true_exponent = t_exponent_field - bias
    end

    # Common path for conversion to BFloat16
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

@generated function BFloat16(x::T) where T<:Microfloat
    lookup = Tuple(to_bfloat16(reinterpret(T, i % UInt8)) for i in 0:2^total_bits(T)-1)
    :($lookup[reinterpret(UInt8, x) + 0x0001])
end

(::Type{T})(x::Microfloat) where {T<:AbstractFloat} = T(BFloat16(x))