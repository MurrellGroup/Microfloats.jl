"""
    @microfloat name [kwargs...]

Declare `name` as a new 8-bit primitive subtype of [`Microfloat`](@ref) and
register its layout, non-finite encoding, and *default* overflow policy.
The default policy is what [`overflow_policy`](@ref) returns; callers
override per-call with the `overflow=` keyword on the conversion.

Default policy rule: [`OVF`](@ref) when the type has any non-finite
sentinel ([`IEEE`](@ref) or [`NanOnlyAllOnes`](@ref)), else [`SAT`](@ref)
(forced for [`FiniteOnly`](@ref) since no sentinel encoding exists).

## Keyword arguments

- `sign`: Number of sign bits: `1` (default) or `0`
- `exponent`: Number of exponent bits: ≥ `1`
- `significand`: Number of significand / mantissa bits: ≥ `0`
- `nonfinite`: [`IEEE`](@ref) (default), [`NanOnlyAllOnes`](@ref), or [`FiniteOnly`](@ref).
- `overflow`: Default overflow policy *instance* — typically [`SAT`](@ref)
  or [`OVF`](@ref). Default: `OVF` if the type has any non-finite values
  (`IEEE`, `NanOnlyAllOnes`), otherwise `SAT` (`FiniteOnly`). Callers can
  override per call with `T(x; overflow=SAT)`.

## Examples

Define a custom 6-bit type with 3 exponent bits, 2 significand bits, and
no Inf/NaN encoding:

```jldoctest
julia> @microfloat MyFloat6 exponent=3 significand=2 nonfinite=Microfloats.FiniteOnly

julia> MyFloat6(1.5)
MyFloat6(1.5)

julia> Microfloats.bitwidth(MyFloat6)
6
```

Conversion from larger types rounds to the nearest representable value,
breaking ties to even (the value whose bit representation ends in `0`):

```jldoctest
julia> Float8_E4M3(0.99)
Float8_E4M3(1.0)
```
"""
macro microfloat(name, kwargs...)
    mod = @__MODULE__
    S = 1
    E = nothing
    M = nothing
    nonfinite = nothing
    overflow = nothing

    for kw in kwargs
        (kw isa Expr && kw.head == :(=)) ||
            error("@microfloat: expected keyword arguments (e.g. exponent=5), got $kw")
        k, v = kw.args
        if k == :sign
            S = Int(v)
        elseif k == :exponent
            E = Int(v)
        elseif k == :significand
            M = Int(v)
        elseif k == :nonfinite
            nonfinite = v
        elseif k == :overflow
            overflow = v
        else
            throw(ArgumentError("@microfloat: unknown keyword `$k`"))
        end
    end

    E === nothing        && error("@microfloat: `exponent` is required")
    M === nothing        && error("@microfloat: `significand` is required")
    S in (0, 1)          || error("@microfloat: `sign` must be 0 or 1, got $S")
    E >= 1               || error("@microfloat: `exponent` must be >= 1, got $E")
    M >= 0               || error("@microfloat: `significand` must be >= 0, got $M")
    S + E + M <= 8       || error("@microfloat: `sign + exponent + significand` must be <= 8, got $(S + E + M)")

    nonfinite_expr = nonfinite === nothing ? :($IEEE) : esc(nonfinite)
    overflow_expr = overflow === nothing ?
        :($hasinf($nonfinite_expr) || $hasnan($nonfinite_expr) ? $OVF : $SAT) :
        esc(overflow)

    T = esc(name)
    N = S + E + M

    quote
        Base.@__doc__ primitive type $T <: $Microfloat 8 end
        $_validate_microfloat($T, $nonfinite_expr, $overflow_expr)
        $mod.float_bits(::Type{$T}) = ($S, $E, $M)
        $mod.non_finite_behavior(::Type{$T}) = $nonfinite_expr
        $mod.overflow_policy(::Type{$T}) = $overflow_expr
        let lookup = Tuple($_to_bfloat16(reinterpret($T, i % UInt8)) for i in 0:$(2^N - 1))
            $mod.to_bfloat16(x::$T) = lookup[reinterpret(UInt8, x) + 0x0001]
        end
        let strings = Tuple($_shortest_decimal_string(reinterpret($T, i % UInt8)) for i in 0:$(2^N - 1))
            $mod.decimal_string(x::$T) = strings[reinterpret(UInt8, x) + 0x0001]
        end
    end
end

function _validate_microfloat(T, nonfinite, overflow)
    (nonfinite isa Type && nonfinite <: NonFiniteBehavior) ||
        throw(ArgumentError("@microfloat($T): `nonfinite` must be IEEE, NanOnlyAllOnes, or FiniteOnly, got $nonfinite"))
    (overflow isa OverflowPolicy) ||
        throw(ArgumentError("@microfloat($T): `overflow` must be an OverflowPolicy instance (e.g. SAT or OVF), got $overflow"))
    nonfinite === FiniteOnly && overflow === OVF &&
        throw(ArgumentError("@microfloat($T): `overflow=OVF` invalid for `nonfinite=FiniteOnly` (no sentinel encoding)"))
    return nothing
end
