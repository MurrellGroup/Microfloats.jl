using BitPacking: NArray

# Hand-optimized bit-twiddling `cvt` specializations.
#
# Every Float4_E2M1FN value is exactly representable in the E4M3 layouts
# (M: 1 ≤ 3; exponent range ⊂ target range; subnormal 0.5 becomes normal),
# so the conversion is independent of rounding mode and overflow policy and
# reduces to a few branch-free ALU ops on the raw bits. Unlike the
# `@cvt_table` lookup (a gather in vectorized loops), these autovectorize.
#
# Pairs implemented here are excluded from the table-registration loop in
# variants.jl (`TWIDDLED_PAIRS`) so the definitions don't collide.

# 3-bit magnitude m = e₁e₀f: 0 → 0; 1 (subnormal, 0.5) → 0x30 (2⁻¹);
# normals are linear: (e+6) << 3 | f << 2 == (m << 2) + 0x30.
@inline function _e2m1_to_e4m3_mag(m::UInt8)
    t = (m << 2) + 0x30
    t = ifelse(m == 0x01, 0x30, t)
    return ifelse(m == 0x00, 0x00, t)
end

@inline _e2m1_to_e4m3_byte(n::UInt8) =
    _e2m1_to_e4m3_mag(n & 0x07) | ((n & 0x08) << 4)

for T in (:Float8_E4M3, :Float8_E4M3FN)
    @eval begin
        @inline cvt(::Type{$T}, x::Float4_E2M1FN,
                    ::RoundingMode, ::OverflowPolicy) =
            reinterpret($T, _e2m1_to_e4m3_byte(reinterpret(UInt8, x)))

        # Packed → packed: twiddle directly on the storage bits, so loops
        # over packed buffers never materialize individual lanes.
        @inline function cvt(::Type{NVector{$T,2}}, xs::NVector{Float4_E2M1FN,2},
                             ::RoundingMode, ::OverflowPolicy)
            raw = reinterpret(UInt8, xs) # lane 1 in the low nibble
            lo = _e2m1_to_e4m3_byte(raw & 0x0f)
            hi = _e2m1_to_e4m3_byte(raw >> 4)
            return NArray{$T,1,Tuple{2},UInt16}(UInt16(lo) | (UInt16(hi) << 8))
        end

        @inline function cvt(::Type{NVector{$T,4}}, xs::NVector{Float4_E2M1FN,4},
                             ::RoundingMode, ::OverflowPolicy)
            raw = reinterpret(UInt16, xs)
            b1 = _e2m1_to_e4m3_byte(raw % UInt8 & 0x0f)
            b2 = _e2m1_to_e4m3_byte((raw >> 4) % UInt8 & 0x0f)
            b3 = _e2m1_to_e4m3_byte((raw >> 8) % UInt8 & 0x0f)
            b4 = _e2m1_to_e4m3_byte((raw >> 12) % UInt8)
            return NArray{$T,1,Tuple{4},UInt32}(
                UInt32(b1) | (UInt32(b2) << 8) | (UInt32(b3) << 16) | (UInt32(b4) << 24))
        end
    end
end
