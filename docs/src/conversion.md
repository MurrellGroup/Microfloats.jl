
## Conversion

### BFloat16

Conversion to and from `Microfloat` uses `BFloat16` as an intermediate type,
since BFloat16 has 1 sign bit, 8 exponent bits, and 7 significand bits,
and is therefore able to represent all `Microfloat` types.

### Rounding

Converting from larger types will round to the nearest even value, i.e.
the value whose bit representation ends in 0.

### Overflow policies

When converting from a wider type to a `Microfloat`, one may want certain behaviors
in regard to Inf and NaN handling.

<table border="1" style="text-align: center;">
<style>
th, td { text-align: center !important; }
</style>
    <thead>
        <tr>
            <th rowspan="3">Source Value <br> (after rounding)</th>
            <th colspan="6">Destination Value</th>
        </tr>
        <tr>
            <th colspan="2">Has Inf+NaN</th>
            <th colspan="2">Has NaN</th>
            <th colspan="2">Finite</th>
        </tr>
        <tr>
            <th>SAT</th>
            <th>OVF</th>
            <th>SAT</th>
            <th>OVF</th>
            <th>SAT</th>
            <th>OVF</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align: left !important;">NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>Error</td>
            <td>Error</td>
        </tr>
        <tr>
            <td style="text-align: left !important;">±Inf</td>
            <td>±floatmax</td>
            <td>±Inf</td>
            <td>±floatmax</td>
            <td>NaN</td>
            <td>±floatmax</td>
            <td>Error</td>
        </tr>
        <tr>
            <td style="text-align: left !important;">>|floatmax|</td>
            <td>±floatmax</td>
            <td>±Inf</td>
            <td>±floatmax</td>
            <td>NaN</td>
            <td>±floatmax</td>
            <td>Error</td>
        </tr>
    </tbody>
</table>

```@docs
OVF
SAT
```
