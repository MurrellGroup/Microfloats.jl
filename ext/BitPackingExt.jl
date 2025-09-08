module BitPackingExt

using Microfloats
using BitPacking

BitPacking.bitwidth(::Type{T}) where T<:Microfloat = Microfloats.n_bits(T)

end
