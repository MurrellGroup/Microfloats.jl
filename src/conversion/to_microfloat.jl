e_subnormal(T) = 1 - bias(T) - n_mantissa_bits(T)
e_normal(T) = 1 - bias(T)
e_overflow(T) = (2^n_exponent_bits(T) - 2) - bias(T) + 1

function create_base_shifttable(::Type{T}) where {T<:Microfloat}

    basetable = Vector{T}(undef, 512)
    shifttable = Vector{UInt8}(undef, 512)

    # shift a 0x1 in the exponent bits created by "significand_mask(Float32) + 0x1"
    # to the first significand bit
    # e_shift_subnorm is 17 for Float8
    e_shift_subnorm = n_mantissa_bits(Float32)-(n_mantissa_bits(T)-1)+e_normal(T)-1

    for i = 0:255                               # all possible exponents for Float32
        e = i - 127                             # subtract Float32 bias
        if e < e_normal(T)                      # Very small numbers map round to zero or subnormal
            basetable[i|0x000+1] = zero(T)
            basetable[i|0x100+1] = -zero(T)
            shifttable[i|0x000+1] = -e+e_shift_subnorm
            shifttable[i|0x100+1] = -e+e_shift_subnorm
        elseif e < e_overflow(T)                # Normal numbers just lose precision
            basebits = (e + Int(bias(T))) << exponent_offset(T)
            basetable[i|0x000+1] = reinterpret(T, UInt8(basebits))
            basetable[i|0x100+1] = reinterpret(T, UInt8(basebits | Int(sign_mask(T))))
            shifttable[i|0x000+1] = n_mantissa_bits(Float32)-n_mantissa_bits(T)
            shifttable[i|0x100+1] = n_mantissa_bits(Float32)-n_mantissa_bits(T)
        elseif e < 128                          # Large numbers map to Infinity
            basetable[i|0x000+1] = inf(T)
            basetable[i|0x100+1] = -inf(T)
            # Use a large shift so mantissa contribution is zero (keeps Inf)
            shifttable[i|0x000+1] = n_mantissa_bits(Float32) + 1
            shifttable[i|0x100+1] = n_mantissa_bits(Float32) + 1
        else                                    # Infinity and NaN's stay Infinity and NaN's
            basetable[i|0x000+1] = inf(T)
            basetable[i|0x100+1] = -inf(T)
            # Also suppress mantissa for Float32 Inf inputs so they remain Inf
            shifttable[i|0x000+1] = n_mantissa_bits(Float32) + 1
            shifttable[i|0x100+1] = n_mantissa_bits(Float32) + 1
        end
    end

    return reinterpret(UInt8, basetable), shifttable
end

@generated function (::Type{T})(x::Float32) where {S,E,M,T<:Microfloat{S,E,M}}
    basetable, shifttable = create_base_shifttable(T)

    quote
        isnan(x) && return nan(T) # TODO retain the significant bits for NaN?
        f = reinterpret(UInt32, x)
    
        # exponent+sign index into 512-entry tables (9 bits), 1-based
        i = (f >> exponent_offset(Float32)) + 1
        @inbounds sh = $shifttable[i]
        f &= mantissa_mask(Float32)
    
        # If `val` is subnormal, the tables are set up to force the
        # result to 0, so the significand has an implicit `1` in the
        # cases we care about.
    
        f |= mantissa_mask(Float32) + 0x1
        m = UInt8(((f >> sh) & UInt32(right_aligned_mantissa_mask(T))) << mantissa_offset(T))
        @inbounds h = ($basetable[i] + m) % UInt8
    
        # rounding
        nextbit = (f >> (sh-1)) & 1
        if nextbit != 0 && (h & exponent_mask(T)) != exponent_mask(T)
            # Round half to even on mantissa LSB, considering mantissa_offset
            mantissa_lsb_is_one = ((h >> mantissa_offset(T)) & 0x01) == 0x01
            lower_bits_nonzero = (f & ((UInt32(1) << (sh-1)) - 1)) != 0
            if mantissa_lsb_is_one || lower_bits_nonzero
                h = h + (UInt8(1) << mantissa_offset(T))
            end
        end
        return reinterpret(T, h)
    end
end
