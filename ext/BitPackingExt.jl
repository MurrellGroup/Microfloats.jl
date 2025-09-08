module BitPackingExt

using Microfloats
using BitPacking

BitPacking.bitwidth(::Type{<:Microfloat}) = Microfloats.n_bits(T)

end
