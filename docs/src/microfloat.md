# Microfloat

```@docs
Microfloat
```

## Variants

### Finite

```@docs
Finite
```

### IEEE 754-like

These types have IEEE 754-like Inf/NaN encodings, with Inf being represented as all 1s in the exponent and a significand of zero, and NaN being represented as all 1s in the exponent and a non-zero significand.

```@docs
IEEE_754_like
Float8_E3M4
Float8_E4M3
Float8_E5M2
Float6_E2M3
Float6_E3M2
Float4_E2M1
```

### Microscaling (MX)

Types from [Open Compute Project Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), with `MX_E5M2` adhering to the IEEE 754-like encoding of Inf/NaN,
whereas MX_E4M3 and MX_E8M0 have no Inf, and only one representation of NaN (excluding the sign bit),
and the finite types MX_E3M2, MX_E2M3, and MX_E2M1 which have no Inf or NaNs.

```@docs
MX
MX_E5M2
MX_E4M3
MX_E3M2
MX_E2M3
MX_E2M1
MX_E8M0
```
