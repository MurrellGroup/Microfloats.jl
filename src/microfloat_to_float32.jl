function Base.Float32(x::T) where T<:Microfloat
    ix = reinterpret(UInt8, x)

    sgn = UInt32(ix & sign_mask(T)) >> 7
    exp  = UInt32(ix & exponent_mask(T)) >> exponent_offset(T)
    mnt = UInt32(ix & mantissa_mask(T)) >> mantissa_offset(T)

    if _onlyfinite(T) && exp == exp_bits_all_one(T)
        return Float32(floatmax(T)) * 2        # same value `inf(T)` clips to
    end

    if exp == 0              # subnormals & zeros
        if mnt == 0
            return reinterpret(Float32, sgn << 31)
        else
        # — this whole block is suspect —
        n_bit = 1
        bit   = first_mantissa_bit_mask(typeof(x))
        while iszero(bit & mnt)
            n_bit += 1
            bit   >>= 1
        end

        # signed‐zero vs normal subnormal exponent:
        sgn = sgn << 31
        exp = ((bias_difference(typeof(x)) - n_bit) << 23) % UInt32

        # and the way you shift mant into the 23-bit fraction:
        mnt = ((mnt & (~bit)) << n_bit) << mantissa_bit_shift(typeof(x))

        return reinterpret(Float32, sgn | exp | mnt)
        end
    elseif exp == exp_bits_all_one(typeof(x))
        if iszero(mnt)
            return ifelse(iszero(sgn), Inf32, -Inf32)
        else
            sgn = sgn << 31
            exp = 0x7fc00000
            mnt = mnt << mantissa_bit_shift(typeof(x))
            return reinterpret(Float32, sgn | exp | mnt)
        end
    else
        sgn = sgn << 31
        exp = (exp + bias_difference(typeof(x))) << 23
        mnt = mnt << mantissa_bit_shift(typeof(x))
        return reinterpret(Float32, sgn | exp | mnt)
    end
end
