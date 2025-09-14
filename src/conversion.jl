
abstract type OverflowPolicy end
struct OVF{T} <: OverflowPolicy; x::T end
struct SAT{T} <: OverflowPolicy; x::T end

function rshift_round_to_even(x::UInt16, n::Int)
    n <= 0 && return x << (-n)
    lower = x & ((UInt16(1) << n) - UInt16(1))
    half = UInt16(1) << (n - 1)
    up = (lower > half) | ((lower == half) & (((x >> n) & UInt16(1)) == UInt16(1)))
    (x >> n) + (up ? UInt16(1) : UInt16(0))
end

function (::Type{T})(x::BFloat16, ::Type{P}=OVF) where {T<:Microfloat,P<:OverflowPolicy}
    bf16_raw = reinterpret(UInt16, x)
    bf16_sign_field = (bf16_raw >> 15) & 0x1
    bf16_exponent_field = Int((bf16_raw >> 7) & 0xff)
    bf16_fraction_field = UInt16(bf16_raw & 0x7f)

    is_negative = (sign_bits(T) == 1) && (bf16_sign_field == 0x1)

    t_exponent_all_ones = (1 << exponent_bits(T)) - 1

    if bf16_exponent_field == 0xff
        if bf16_fraction_field == 0x00
            y = P <: SAT ? floatmax(T) : inf(T)
            return is_negative ? -y : y
        else
            if significand_bits(T) == 0
                y = P <: SAT ? floatmax(T) : inf(T)
                return is_negative ? -y : y
            else
                t_raw = UInt8(((t_exponent_all_ones & ((1 << exponent_bits(T)) - 1)) << significand_bits(T)) & 0xff)
                t_raw |= UInt8(0x01)
                if sign_bits(T) == 1 && is_negative
                    t_raw |= UInt8(1) << (exponent_bits(T) + significand_bits(T))
                end
                return reinterpret(T, t_raw)
            end
        end
    end

    bf16_significand_8bit = bf16_exponent_field == 0 ? UInt16(bf16_fraction_field) : (UInt16(0x80) | UInt16(bf16_fraction_field))
    bf16_true_exponent = bf16_exponent_field == 0 ? -126 : (bf16_exponent_field - 127)

    t_exponent_field = bf16_true_exponent + exponent_bias(T)

    if t_exponent_field <= 0
        shift_to_subnormal = t_exponent_field + significand_bits(T) - 8
        sub_q = rshift_round_to_even(bf16_significand_8bit, -shift_to_subnormal)
        max_fraction_field = UInt16((1 << significand_bits(T)) - 1)
        if sub_q == 0
            z = zero(T)
            return (is_negative && sign_bits(T) == 1) ? -z : z
        elseif sub_q == (UInt16(1) << significand_bits(T))
            t_raw = UInt8(0)
            t_raw |= UInt8(1) << significand_bits(T)
            if sign_bits(T) == 1 && is_negative
                t_raw |= UInt8(1) << (exponent_bits(T) + significand_bits(T))
            end
            return reinterpret(T, t_raw)
        else
            sub_q = min(sub_q, max_fraction_field)
            t_raw = UInt8(sub_q & max_fraction_field)
            if sign_bits(T) == 1 && is_negative
                t_raw |= UInt8(1) << (exponent_bits(T) + significand_bits(T))
            end
            return reinterpret(T, t_raw)
        end
    else
        shift_to_t_fraction = 7 - significand_bits(T)
        t_significand_total = shift_to_t_fraction >= 0 ? rshift_round_to_even(bf16_significand_8bit, shift_to_t_fraction) : (bf16_significand_8bit << (-shift_to_t_fraction))
        t_exponent_field_rounded = t_exponent_field
        if t_significand_total == (UInt16(1) << (significand_bits(T) + 1))
            t_significand_total = UInt16(1) << significand_bits(T)
            t_exponent_field_rounded += 1
        end
        if t_exponent_field_rounded >= t_exponent_all_ones
            y = P <: SAT ? floatmax(T) : inf(T)
            return is_negative ? -y : y
        end
        t_fraction_field = UInt8((t_significand_total - (UInt16(1) << significand_bits(T))) & UInt16((1 << significand_bits(T)) - 1))
        t_raw = UInt8(0)
        t_raw |= UInt8(t_exponent_field_rounded & ((1 << exponent_bits(T)) - 1)) << significand_bits(T)
        t_raw |= t_fraction_field
        if sign_bits(T) == 1 && is_negative
            t_raw |= UInt8(1) << (exponent_bits(T) + significand_bits(T))
        end
        return reinterpret(T, t_raw)
    end
end

(::Type{T})(x, args...) where {T<:Microfloat} = T(BFloat16(x), args...)
(::Type{T})(wrapper::P) where {T<:Microfloat,P<:OverflowPolicy} = T(wrapper.x, P)

function BFloat16(x::T) where {T<:Microfloat}
    t_raw = reinterpret(UInt8, x)

    t_sign = (sign_bits(T) == 1) && (t_raw & (UInt8(1) << (exponent_bits(T) + significand_bits(T))) != 0)
    t_exponent_field = Int((t_raw >> significand_bits(T)) & UInt8((1 << exponent_bits(T)) - 1))
    t_fraction_field = UInt16(t_raw & UInt8((1 << significand_bits(T)) - 1))

    bf16_sign_bit = UInt16(t_sign ? 1 : 0) << 15

    if t_exponent_field == ((1 << exponent_bits(T)) - 1)
        if t_fraction_field == 0
            return reinterpret(BFloat16, bf16_sign_bit | 0x7f80)
        else
            return reinterpret(BFloat16, bf16_sign_bit | 0x7fc0)
        end
    elseif t_exponent_field == 0
        if t_fraction_field == 0
            return reinterpret(BFloat16, bf16_sign_bit)
        else
            t_significand_total = t_fraction_field
            t_true_exponent = 1 - exponent_bias(T)
            shift_to_bf16_sub = t_true_exponent + 133 - significand_bits(T)
            sub_q = shift_to_bf16_sub >= 0 ? (t_significand_total << shift_to_bf16_sub) : rshift_round_to_even(t_significand_total, -shift_to_bf16_sub)
            if sub_q == 0
                return reinterpret(BFloat16, bf16_sign_bit)
            elseif sub_q >= 0x80
                return reinterpret(BFloat16, bf16_sign_bit | UInt16(0x0080))
            else
                return reinterpret(BFloat16, bf16_sign_bit | UInt16(sub_q & 0x7f))
            end
        end
    else
        t_significand_total = (UInt16(1) << significand_bits(T)) + t_fraction_field
        t_true_exponent = t_exponent_field - exponent_bias(T)
        bf16_exponent_field = t_true_exponent + 127
        bf16_significand_total = if significand_bits(T) >= 7
            rshift_round_to_even(t_significand_total, significand_bits(T) - 7)
        else
            t_significand_total << (7 - significand_bits(T))
        end
        if bf16_significand_total == UInt16(0x100)
            bf16_significand_total = UInt16(0x80)
            bf16_exponent_field += 1
        end
        if bf16_exponent_field >= 0xff
            return reinterpret(BFloat16, bf16_sign_bit | 0x7f80)
        elseif bf16_exponent_field <= 0
            shift_to_bf16_sub = t_true_exponent + 133 - significand_bits(T)
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
end

(::Type{T})(x::Microfloat) where {T<:AbstractFloat} = T(BFloat16(x))