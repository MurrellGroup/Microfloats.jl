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
