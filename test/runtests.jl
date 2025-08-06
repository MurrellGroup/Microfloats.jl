using Microfloats
using Test

using Microfloats: n_bits, n_sign_bits, n_exponent_bits, n_mantissa_bits, exponent_offset, mantissa_offset, sign_mask, exponent_mask, mantissa_mask, _onlyfinite
using Microfloats, Test
using Microfloats, Test

# —— Helpers —— #

# exact power-of-two as a Rational, handling negative exponents correctly
pow2(n::Integer) = n ≥ 0 ? BigInt(1) << n // BigInt(1) :
                         BigInt(1) // (BigInt(1) << -n)

"""
    ref_rational_value(::Type{T}, bits8)

Interpret the byte `bits8` as a Microfloat of type `T` and return its
exact value as a Rational{BigInt}.  (Only call on finite patterns.)
"""
function ref_rational_value(::Type{T}, bits8::UInt8) where T<:Microfloat
    # extract fields
    sign = (bits8 & sign_mask(T)) != 0 ? -1 : +1
    exp  = Int((bits8 & exponent_mask(T)) >> exponent_offset(T))
    mant = BigInt((bits8 & mantissa_mask(T)) >> mantissa_offset(T))

    EB   = n_exponent_bits(T)
    MB   = n_mantissa_bits(T)
    bias = Int(Microfloats.bias(T))
    maxE = (1 << EB) - 1

    if exp == 0
        if mant == 0
            return 0//1
        else
            # subnormal: mant/2^MB * 2^(1 - bias)
            return sign * (mant // (BigInt(1) << MB)) * pow2(1 - bias)
        end
    elseif exp < maxE
        # normal: (2^MB + mant)/2^MB * 2^(exp - bias)
        num = (BigInt(1) << MB) + mant
        den = BigInt(1) << MB
        return sign * (num // den) * pow2(exp - bias)
    else
        error("ref_rational_value called on non-finite pattern")
    end
end

"""
    ref_to_Float32(x::T) where T<:Microfloat

Convert x to its *correct* IEEE-754 Float32 by first forming the exact
Rational value and then `convert(Float32, ::Rational)`, which does
round-to-even.
"""
ref_to_Float32(x::T) where T<:Microfloat =
    convert(Float32, ref_rational_value(typeof(x), reinterpret(UInt8, x)))

"""
    ref_pattern_from_Float32(f::Float32, ::Type{T}) where T<:Microfloat

Brute-force over all 2^N patterns of T, for each non-∞ target compute
its reference Float32 via `ref_to_Float32`, and pick the pattern whose
result is nearest to `f` (tie → even LSB).  Returns the correct
T-value for that pattern.
"""
function ref_pattern_from_Float32(f::Float32, ::Type{T}) where T<:Microfloat
    N = n_bits(T)
    best_i, best_diff = UInt8(0), nothing
    for i in UInt8.(0:UInt8(2^N-1))
        bits8 = i << (8 - N)
        t     = reinterpret(T, bits8)
        # only consider finite targets
        f_ref = isfinite(t) ? ref_to_Float32(t) : Float32(t)
        d     = abs(Float64(f_ref) - Float64(f))
        if best_diff === nothing ||
           d <  best_diff ||
          (d == best_diff && iseven(Int(i)))
            best_diff = d
            best_i    = i
        end
    end
    return reinterpret(T, best_i << (8 - N))
end


# —— The Tests —— #

@testset "bit-perfect Microfloat ↔ Float32" begin
  for OnlyFinite in (false, true), N in 1:8, S in 0:1, E in 1:(N - S)
    M = N - S - E
    T = Microfloat{N,E,M,S,OnlyFinite}

    @testset "$T ↔ Float32" begin

      # 1) to-Float32: every pattern
      for raw in UInt8.(0:UInt8(2^N-1))
        bits8 = raw << (8 - N)
        x     = reinterpret(T, bits8)
        f     = Float32(x)

        if isfinite(x)
          @test f === ref_to_Float32(x)
        else
          # special: finite-only saturates; IEEE style splits Inf/Nan by mantissa
          if OnlyFinite
            @test isfinite(f)
          else
            if (bits8 & mantissa_mask(T)) == 0
              @test isinf(f)
            else
              @test isnan(f)
            end
          end
        end
      end

      # 2) from-Float32: every finite output
      for raw in UInt8.(0:UInt8(2^N-1))
        bits8 = raw << (8 - N)
        x     = reinterpret(T, bits8)
        f     = Float32(x)

        if isfinite(f)
          t_impl = T(f)
          t_ref  = ref_pattern_from_Float32(f, T)
          @test t_impl === t_ref
        else
          # NaN → NaN, Inf → typemax/typemin
          if isnan(f)
            @test isnan(T(f))
          else
            @test T(f) == (f < 0 ? typemin(T) : typemax(T))
          end
        end
      end

    end
  end
end