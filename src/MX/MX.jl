# src/MX/MX.jl

const MXMicrofloat{S,E,M} = Microfloat{S,E,M,:MX}
const SignedMXMicrofloat = MXMicrofloat{1}
const UnsignedMXMicrofloat = MXMicrofloat{0}

# E8M0 (scale type)
Base.isnan(x::T) where T<:MXMicrofloat{0,8,0} = reinterpret(UInt8, x) == 0xff
nan(::Type{T}) where T<:MXMicrofloat{0,8,0} = reinterpret(T, 0xff)
Base.isinf(::T) where T<:MXMicrofloat{0,8,0} = false

# E4M3 (MX): no Infs; only mantissa == 111 at exp=1111 is NaN
nan(::Type{T}) where {S,T<:MXMicrofloat{S,4,3}} = reinterpret(T, exponent_mask(T) | mantissa_mask(T))
Base.isnan(x::T) where {S,T<:MXMicrofloat{S,4,3}} =
    (only_exponent(x) == exponent_mask(T)) && (only_mantissa(x) == mantissa_mask(T))
Base.isinf(::T) where {S,T<:MXMicrofloat{S,4,3}} = false
Base.floatmax(::Type{T}) where {S,T<:MXMicrofloat{S,4,3}} =
    reinterpret(T, exponent_mask(T) | (mantissa_mask(T) & ~(bit_ones(1, uint(T)) << mantissa_offset(T))))

# no adjustments for E5M2

# E3M2 (MX): no Infs, no NaN sentinel; NaN maps to max finite
nan(::Type{T}) where {S,T<:MXMicrofloat{S,3,2}} = floatmax(T)
Base.isnan(::T) where {S,T<:MXMicrofloat{S,3,2}} = false
Base.isinf(::T) where {S,T<:MXMicrofloat{S,3,2}} = false
Base.floatmax(::Type{T}) where {S,T<:MXMicrofloat{S,3,2}} =
    reinterpret(T, exponent_mask(T) | mantissa_mask(T))

# E2M3 (MX): no Infs, no NaN sentinel; NaN maps to max finite
nan(::Type{T}) where {S,T<:MXMicrofloat{S,2,3}} = floatmax(T)
Base.isnan(::T) where {S,T<:MXMicrofloat{S,2,3}} = false
Base.isinf(::T) where {S,T<:MXMicrofloat{S,2,3}} = false
Base.floatmax(::Type{T}) where {S,T<:MXMicrofloat{S,2,3}} =
    reinterpret(T, exponent_mask(T) | mantissa_mask(T))

# E2M1 (MX): no Infs, no NaN sentinel; NaN maps to max finite
nan(::Type{T}) where {S,T<:MXMicrofloat{S,2,1}} = floatmax(T)
Base.isnan(::T) where {S,T<:MXMicrofloat{S,2,1}} = false
Base.isinf(::T) where {S,T<:MXMicrofloat{S,2,1}} = false
Base.floatmax(::Type{T}) where {S,T<:MXMicrofloat{S,2,1}} =
    reinterpret(T, exponent_mask(T) | mantissa_mask(T))

# Float32 conversion for MX variants:
# - exp=all-ones is "normal" except for the MX NaN sentinel(s)
# - otherwise identical mapping as IEEE
@inline function _float32(x::T) where {T<:MXMicrofloat}
    early = _float32_injection(x)
    isnothing(early) || return early

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
    is_scale = (n_sign_bits(T) == 0 && n_exponent_bits(T) == 8 && n_mantissa_bits(T) == 0)
    e_overflow_mx = is_scale ? Int(e_overflow(T)) : Int(e_overflow(T)) + 1

    for i = 0:255
        e = i - 127
        if e < e_subnormal(T)
            basetable[i|0x000+1] = zero(T)
            basetable[i|0x100+1] = -zero(T)
            # Provide a large shift so rounding logic can raise to the minimal subnormal when appropriate
            sh = -e + e_shift_subnorm
            shifttable[i|0x000+1] = sh
            shifttable[i|0x100+1] = sh
        elseif e < e_normal(T)
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

# E8M0 specific early injection remains; fill in more if needed
function _float32_injection(x::T) where T<:UnsignedMXMicrofloat{8,0}
    reinterpret(UInt8, x) == 0xff && return NaN32
    nothing
end

# Saturating bounds for MX: use finite extrema
Base.typemax(::Type{T}) where {S,E,M,T<:MXMicrofloat{S,E,M}} = floatmax(T)
Base.typemin(::Type{T}) where {S,E,M,T<:MXMicrofloat{S,E,M}} = ifelse(n_sign_bits(T) == 0, zero(T), -floatmax(T))