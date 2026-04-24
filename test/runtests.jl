using Microfloats
using Microfloats: non_finite_behavior, hasinf, hasnan, inf, nan,
                   sign_bits, bitwidth, overflow_policy,
                   IEEE, NanOnlyAllOnes, FiniteOnly,
                   OverflowPolicy, SAT, OVF,
                   @microfloat
using Test
using Random

using BFloat16s: BFloat16

# NaN-aware bit-identity for tests. Overrides `===` for Microfloat pairs so
# round-trip tests can treat distinct NaN encodings as equivalent without
# losing the bit-identity semantics for signed zeros.
≡(a, b) = isnan(a) || isnan(b) ? isnan(a) && isnan(b) : a == b

# variants used only for tests
@microfloat UFloat7_E3M4  sign=0 exponent=3 significand=4
@microfloat UFloat7_E4M3  sign=0 exponent=4 significand=3
@microfloat UFloat7_E5M2  sign=0 exponent=5 significand=2
@microfloat UFloat7_E4M3FN sign=0 exponent=4 significand=3 nonfinite=NanOnlyAllOnes
@microfloat UFloat5_E2M3  sign=0 exponent=2 significand=3 nonfinite=FiniteOnly
@microfloat UFloat5_E3M2  sign=0 exponent=3 significand=2 nonfinite=FiniteOnly
@microfloat UFloat3_E2M1  sign=0 exponent=2 significand=1 nonfinite=FiniteOnly

# Twin of Float8_E4M3FN with the alternate (PyTorch/Triton) overflow policy.
# Demonstrates the documented "reinterpret between twin types" escape hatch;
# used in overflow.jl.
@microfloat _E4M3FN_SAT exponent=4 significand=3 nonfinite=NanOnlyAllOnes overflow=SAT

const SIGNED_TYPES = (
    Float8_E3M4, Float8_E4M3, Float8_E5M2,
    Float8_E4M3FN,
    Float6_E2M3FN, Float6_E3M2FN,
    Float4_E2M1FN,
)

const UNSIGNED_TYPES = (
    Float8_E8M0FNU,
    UFloat7_E3M4, UFloat7_E4M3, UFloat7_E5M2,
    UFloat7_E4M3FN,
    UFloat5_E2M3, UFloat5_E3M2,
    UFloat3_E2M1,
)

const TYPES = (SIGNED_TYPES..., UNSIGNED_TYPES...)

# OCP Microscaling Formats v1.0 aliases
const MX_E5M2 = Float8_E5M2
const MX_E4M3 = Float8_E4M3FN
const MX_E3M2 = Float6_E3M2FN
const MX_E2M3 = Float6_E2M3FN
const MX_E2M1 = Float4_E2M1FN
const MX_E8M0 = Float8_E8M0FNU

@testset "Microfloats" begin
    include("basic.jl")
    include("overflow.jl")
    include("mx_compliance.jl")
    include("mx_properties.jl")
    include("dlfp8_parity.jl")
end
