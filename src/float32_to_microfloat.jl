function (::Type{T})(x::Float32) where {N,E,M,S,T<:Microfloat{N,E,M,S,true}}
    if !isfinite(x)
        return x < 0 ? typemin(T) : typemax(T)
    else
        return reinterpret(T, Microfloat{N,E,M,S,false}(x))
    end
end


@inline _implicit_one(mbits) = one(UInt32) << mbits          # 1 << 23 etc.

@inline function rne_round(trunc::T, full::T, shift::Int) where {T<:Unsigned}
    guard      = (full >> (shift-1)) & one(T)
    round_bits = full & ((one(T) << (shift-1)) - one(T))

    return (guard == one(T) &&
            (round_bits != zero(T) || (trunc & one(T)) == one(T))) ?
           trunc + one(T) : trunc
end


function (::Type{T})(x::Float32) where {N,E,M,S,T<:Microfloat{N,E,M,S,false}}
    EB  = n_exponent_bits(T)
    MB  = n_mantissa_bits(T)

    bits32  = reinterpret(UInt32, x)
    sign32  = (bits32 >> 31) & 0x01
    exp32   = (bits32 >> 23) & 0xff
    mant32  = bits32 & 0x7fffff          # already UInt32

    bias32  = UInt32(127)
    biasT   = bias(T)
    maxExpT = (one(UInt32) << EB) - one(UInt32)

    expT  = UInt32(0)
    mantT = UInt32(0)

    if exp32 == 0xff
        expT  = maxExpT
        mantT = iszero(mant32) ? 0 :
                max(UInt32(1), mant32 >> (23-MB))

    elseif exp32 == 0x00
        if mant32 == 0
            expT = 0            # ±0
        else
            lead     = leading_zeros(mant32) - 9
            mant32 <<= (lead + 1)
            expunb   = Int32(1 - Int32(bias32) - lead - 1)
            expunbT  = expunb + Int32(biasT)

            if expunbT ≤ 0
                shift = (1 - expunbT) + (23 - MB)
                total  = _implicit_one(23) | mant32
                mantT  = rne_round(total >> shift, total, shift)
                expT   = 0
            else
                expT   = UInt32(expunbT)
                mantT  = rne_round(mant32 >> (23 - MB), mant32, 23 - MB)
            end
        end
    else
        expunb  = Int32(exp32) - Int32(bias32)
        expunbT = expunb + Int32(biasT)

        if expunbT ≥ Int32(maxExpT)
            expT = maxExpT          # overflow → saturate to maxExp
        elseif expunbT ≤ 0
            shift = (1 - expunbT) + (23 - MB)
            total  = _implicit_one(23) | mant32
            mantT  = rne_round(total >> shift, total, shift)
            expT   = 0
        else
            expT  = UInt32(expunbT)
            mantT = rne_round(mant32 >> (23 - MB), mant32, 23 - MB)
        end
    end

    bits8 = UInt8( (S == 0 ? 0 : sign32 << 7) |
                   (expT  << exponent_offset(T)) |
                   (mantT << mantissa_offset(T)) )

    return reinterpret(T, bits8)
end