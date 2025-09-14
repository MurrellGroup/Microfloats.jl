module BitPackingExt

using Microfloats
using BitPacking

BitPacking.bitwidth(::Type{T}) where T<:Microfloat = Microfloats.total_bits(T)

end
