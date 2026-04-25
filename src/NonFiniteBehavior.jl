abstract type NonFiniteBehavior end

"""
IEEE-754-style non-finite encoding: an all-ones exponent with `significand == 0`
encodes `±Inf`, and an all-ones exponent with `significand ≠ 0` encodes `NaN`.
Compatible with both [`OVF`](@ref) and [`SAT`](@ref).

See also [`NanOnlyAllOnes`](@ref), [`FiniteOnly`](@ref).

# Examples
```jldoctest
julia> @microfloat IEEEFloat8 exponent=5 significand=2 nonfinite=Microfloats.IEEE

julia> isinf(IEEEFloat8(Inf)), isnan(IEEEFloat8(NaN))
(true, true)
```
"""
abstract type IEEE           <: NonFiniteBehavior end

"""
NaN is the unique all-ones bit pattern across exponent and significand (per
sign); no `Inf` encoding. The slot that would otherwise be `Inf` is reclaimed
for a finite value, extending dynamic range by one step. Compatible with both
[`OVF`](@ref) and [`SAT`](@ref); under `OVF`, overflow maps to `NaN`.

See also [`IEEE`](@ref), [`FiniteOnly`](@ref).

# Examples
```jldoctest
julia> @microfloat FNFloat8 exponent=4 significand=3 nonfinite=Microfloats.NanOnlyAllOnes

julia> Microfloats.hasinf(FNFloat8), Microfloats.hasnan(FNFloat8)
(false, true)
```
"""
abstract type NanOnlyAllOnes <: NonFiniteBehavior end

"""
No `Inf` or `NaN` — every bit pattern is a finite value, maximizing dynamic
range. Requires `overflow=`[`SAT`](@ref), since there is no sentinel encoding
to represent overflow.

See also [`IEEE`](@ref), [`NanOnlyAllOnes`](@ref).

# Examples
```jldoctest
julia> @microfloat FiniteFloat6 exponent=3 significand=2 nonfinite=Microfloats.FiniteOnly

julia> Microfloats.hasinf(FiniteFloat6), Microfloats.hasnan(FiniteFloat6)
(false, false)
```
"""
abstract type FiniteOnly     <: NonFiniteBehavior end

hasinf(::Type{IEEE})           = true
hasinf(::Type{NanOnlyAllOnes}) = false
hasinf(::Type{FiniteOnly})     = false

hasnan(::Type{IEEE})           = true
hasnan(::Type{NanOnlyAllOnes}) = true
hasnan(::Type{FiniteOnly})     = false
