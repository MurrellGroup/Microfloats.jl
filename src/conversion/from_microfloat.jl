first_mantissa_bit_mask(::Type{T}) where T<:Microfloat = one(UInt32) << (n_mantissa_bits(T) - 1)
mantissa_bit_shift(::Type{T}) where T<:Microfloat = 23 - n_mantissa_bits(T)
exp_bits_all_one(::Type{T}) where T<:Microfloat = UInt32(right_aligned_exponent_mask(T))

function _float32(x::T) where T<:Microfloat
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
        if iszero(mnt)
            return ifelse(iszero(sgn), Inf32, -Inf32)
        else
            sgn = sgn << 31
            exp = 0x7fc00000
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

@generated function Base.Float32(x::T) where {S,E,M,T<:Microfloat{S,E,M}}
    lookup_table = [_float32(reinterpret(T, i)) for i in 0x00:0xff]
    quote
        return $lookup_table[reinterpret(UInt8, x) + 1]
    end
end