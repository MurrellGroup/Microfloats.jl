"""
    @microfloat Name sign=1 exponent=E significand=M nonfinite=Trait

Declare an 8-bit `primitive type Name <: Microfloat{sign,E,M} 8 end` and
register its [`non_finite_behavior`](@ref) as `Trait`.

All keyword arguments are passed positionally as `name=value` pairs:

- `sign`        — `0` or `1`. Default `1`.
- `exponent`    — required, `≥ 1`.
- `significand` — required, `≥ 0`.
- `nonfinite`   — required. One of `IEEE`, `NanOnlyAllOnes`, `FiniteOnly`.

Hand-written equivalent (also supported):

```julia
primitive type Name <: Microfloat{sign,exponent,significand} 8 end
non_finite_behavior(::Type{Name}) = Trait
```
"""
macro microfloat(name, kwargs...)
    mod = @__MODULE__
    S = 1
    E = nothing
    M = nothing
    behavior = IEEE

    for kw in kwargs
        (kw isa Expr && kw.head == :(=)) ||
            error("@microfloat: expected keyword arguments (e.g. exponent=5), got $kw")
        k, v = kw.args
        if k == :sign
            S = v
        elseif k == :exponent
            E = v
        elseif k == :significand
            M = v
        elseif k == :nonfinite
            behavior = v
        else
            error("@microfloat: unknown keyword `$k`")
        end
    end

    E === nothing        && error("@microfloat: `exponent` is required")
    M === nothing        && error("@microfloat: `significand` is required")
    behavior isa NonFiniteBehavior && error("@microfloat: `nonfinite` is required")
    S in (0, 1)          || error("@microfloat: `sign` must be 0 or 1, got $S")
    E >= 1               || error("@microfloat: `exponent` must be >= 1, got $E")
    M >= 0               || error("@microfloat: `significand` must be >= 0, got $M")
    S + E + M <= 8       || error("@microfloat: `sign + exponent + significand` must be <= 8, got $(S + E + M)")

    T = esc(name)
    trait = esc(behavior)
    N = S + E + M
    quote
        primitive type $T <: $mod.Microfloat{$S,$E,$M} 8 end
        $mod.non_finite_behavior(::Type{$T}) = $trait
        let lookup = Tuple($mod._to_bfloat16(reinterpret($T, i % UInt8)) for i in 0:$(2^N - 1))
            $mod.to_bfloat16(x::$T) = lookup[reinterpret(UInt8, x) + 0x0001]
        end
    end
end
