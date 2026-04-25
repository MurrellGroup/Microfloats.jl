# Predefined Microfloats

Microfloats defines and exports a set of common types.

## IEEE-like

These types have IEEE 754-like Inf/NaN encodings, with Inf being represented as all 1s in the exponent and a significand of zero, and NaN being represented as all 1s in the exponent and a non-zero significand.

```@docs
Float8_E5M2
Float8_E4M3
Float8_E3M4
```

## Finite

These types have no Inf encoding, with alternate or no NaN encodings at all.

```@docs
Float8_E4M3FN
Float8_E8M0FNU
Float6_E3M2FN
Float6_E2M3FN
Float4_E2M1FN
```
