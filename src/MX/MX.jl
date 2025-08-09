# src/MX/MX.jl

const MXMicrofloat{S,E,M} = Microfloat{S,E,M,:MX}
const SignedMXMicrofloat = MXMicrofloat{1}
const UnsignedMXMicrofloat = MXMicrofloat{0}

const MX_E5M2 = MXMicrofloat{1,5,2}
const MX_E4M3 = MXMicrofloat{1,4,3}
const MX_E3M2 = MXMicrofloat{1,3,2}
const MX_E2M3 = MXMicrofloat{1,2,3}
const MX_E2M1 = MXMicrofloat{1,2,1}
const MX_E8M0 = MXMicrofloat{0,8,0}

const MX_NO_INF = Union{MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1, MX_E8M0}
const MX_NO_NAN = Union{MX_E3M2, MX_E2M3, MX_E2M1}
const MX_NO_NAN_OR_INF = Union{MX_E3M2, MX_E2M3, MX_E2M1}

Base.isinf(::MX_NO_INF) = false
Base.isnan(::MX_NO_NAN) = false
nan(::Type{T}) where T<:MX_NO_NAN = throw(DomainError(T, "$T has no NaN values"))

Base.floatmax(::Type{T}) where T<:MX_NO_NAN_OR_INF = reinterpret(T, exponent_mask(T) | mantissa_mask(T))

# E4M3 (MX): no Infs; only mantissa == 111 at exp=1111 is NaN
nan(::Type{T}) where T<:MX_E4M3 = reinterpret(T, exponent_mask(T) | mantissa_mask(T))
Base.isnan(x::T) where T<:MX_E4M3 =
    (only_exponent(x) == exponent_mask(T)) && (only_mantissa(x) == mantissa_mask(T))
Base.floatmax(::Type{T}) where T<:MX_E4M3 =
    reinterpret(T, exponent_mask(T) | (mantissa_mask(T) & ~(bit_ones(1, uint(T)) << mantissa_offset(T))))

# E8M0 (scale type)
Base.isnan(x::MX_E8M0) = reinterpret(UInt8, x) == 0xff
nan(::Type{MX_E8M0}) = reinterpret(MX_E8M0, 0xff)

# Float32 conversion for MX variants:
# - exp=all-ones is "normal" except for the MX NaN sentinel(s)
# - otherwise identical mapping as IEEE
function _float32(x::T) where {T<:MXMicrofloat}
    T isa MX_E8M0 && reinterpret(UInt8, x) == 0xff && return NaN32

    sgn = UInt32(right_aligned_sign(x))
    exp = UInt32(right_aligned_exponent(x))
    mnt = UInt32(right_aligned_mantissa(x))

    if exp == 0
        if mnt == 0
            return reinterpret(Float32, sgn << 31)
        else
            n_bit = 1
            bit   = first_mantissa_bit_mask(T)
            while iszero(bit & mnt)
                n_bit += 1
                bit   >>= 1
            end
            sgn = sgn << 31
            exp = ((bias_difference(T) + 1 - n_bit) << 23) % UInt32
            mnt = ((mnt & (~bit)) << n_bit) << mantissa_bit_shift(T)
            return reinterpret(Float32, sgn | exp | mnt)
        end
    elseif exp == exp_bits_all_one(T)
        if isnan(x)
            return NaN32
        else
            sgn = sgn << 31
            exp = (exp + bias_difference(T)) << 23
            mnt = mnt << mantissa_bit_shift(T)
            return reinterpret(Float32, sgn | exp | mnt)
        end
    else
        sgn = sgn << 31
        exp = (exp + bias_difference(T)) << 23
        mnt = mnt << mantissa_bit_shift(T)
        return reinterpret(Float32, sgn | exp | mnt)
    end
end

# Saturating to_microfloat tables for MX (no Infs; overflow -> ±floatmax)
function create_base_shifttable(::Type{T}) where {T<:MXMicrofloat}
    basetable = Vector{T}(undef, 512)
    shifttable = Vector{UInt8}(undef, 512)

    e_shift_subnorm = n_mantissa_bits(Float32) - (n_mantissa_bits(T) - 1) + e_normal(T) - 1
    # MX uses the all-ones exponent as finite for data types (E5M2/E4M3/E3M2/E2M3/E2M1)
    e_overflow_mx = T <: MX_E8M0 ? Int(e_overflow(T)) : Int(e_overflow(T)) + 1

    for i = 0:255
        e = i - 127
        if e < e_normal(T)
            basetable[i|0x000+1] = zero(T)
            basetable[i|0x100+1] = -zero(T)
            shifttable[i|0x000+1] = -e + e_shift_subnorm
            shifttable[i|0x100+1] = -e + e_shift_subnorm
        elseif e < e_overflow_mx
            basebits = (e + Int(bias(T))) << exponent_offset(T)
            basetable[i|0x000+1] = reinterpret(T, UInt8(basebits))
            basetable[i|0x100+1] = reinterpret(T, UInt8(basebits | Int(sign_mask(T))))
            shifttable[i|0x000+1] = n_mantissa_bits(Float32) - n_mantissa_bits(T)
            shifttable[i|0x100+1] = n_mantissa_bits(Float32) - n_mantissa_bits(T)
        elseif e < 128
            basetable[i|0x000+1] = floatmax(T)
            basetable[i|0x100+1] = -floatmax(T)
            shifttable[i|0x000+1] = n_mantissa_bits(T)+1
            shifttable[i|0x100+1] = n_mantissa_bits(T)+1
        else
            basetable[i|0x000+1] = floatmax(T)
            basetable[i|0x100+1] = -floatmax(T)
            shifttable[i|0x000+1] = n_mantissa_bits(Float32) - n_mantissa_bits(T)
            shifttable[i|0x100+1] = n_mantissa_bits(Float32) - n_mantissa_bits(T)
        end
    end
    return reinterpret(UInt8, basetable), shifttable
end

# Saturating bounds for MX: use finite extrema
Base.typemax(::Type{T}) where {S,E,M,T<:MXMicrofloat{S,E,M}} = floatmax(T)
Base.typemin(::Type{T}) where {S,E,M,T<:MXMicrofloat{S,E,M}} = ifelse(n_sign_bits(T) == 0, zero(T), -floatmax(T))
