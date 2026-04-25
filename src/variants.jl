# All shipped types default to `overflow=OVF` if they have any non-finite
# sentinel (IEEE or NanOnlyAllOnes), else forced `SAT` (FiniteOnly has none).
# Rule: `hasinf(T) || hasnan(T) ? OVF : SAT`. Matches cutile-python's
# `_convert_nonfinite` exactly. For PyTorch/Triton-style saturating E4M3FN,
# declare a twin with `overflow=SAT`.

# IEEE-like
@microfloat Float8_E5M2    exponent=5 significand=2
@microfloat Float8_E4M3    exponent=4 significand=3
@microfloat Float8_E3M4    exponent=3 significand=4

# NanOnlyAllOnes (FN-suffixed)
@microfloat Float8_E4M3FN  exponent=4 significand=3 nonfinite=NanOnlyAllOnes
@microfloat Float8_E8M0FNU sign=0 exponent=8 significand=0 nonfinite=NanOnlyAllOnes

# FiniteOnly
@microfloat Float6_E2M3FN  exponent=2 significand=3 nonfinite=FiniteOnly
@microfloat Float6_E3M2FN  exponent=3 significand=2 nonfinite=FiniteOnly
@microfloat Float4_E2M1FN  exponent=2 significand=1 nonfinite=FiniteOnly

for T in (
    :Float8_E5M2, :Float8_E4M3, :Float8_E3M4,
    :Float8_E4M3FN, :Float8_E8M0FNU,
    :Float6_E2M3FN, :Float6_E3M2FN,
    :Float4_E2M1FN,
)
    @eval @doc """
        $($T)

    ## Properties
    - Bits: `$(sign_bits($T))` sign + `$(exponent_bits($T))` exponent + `$(significand_bits($T))` significand (`$(bitwidth($T))` total)
    - Has Inf: `$(hasinf($T))`
    - Has NaN: `$(hasnan($T))`
    - Non-finite behavior: `$(non_finite_behavior($T))`
    - Overflow policy: `$(overflow_policy($T))`
    - Max normal: `$(Float64(floatmax($T)))`
    - Min normal: `$(Float64(floatmin($T)))`
    - Max subnormal: `$(significand_bits($T) > 0 ? Float64(prevfloat(floatmin($T))) : "N/A")`
    - Min subnormal: `$(significand_bits($T) > 0 ? Float64(nextfloat(zero($T))) : "N/A")`
    """ $T
end
