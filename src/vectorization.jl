using BitPacking: NVector
using StaticArrays: SVector

# 16-bit

const Float16x2 = SVector{2,Float16}
const Float16x4 = SVector{4,Float16}

const BFloat16x2 = SVector{2,BFloat16}
const BFloat16x4 = SVector{4,BFloat16}

# 8-bit

const Float8x2_E4M3FN = NVector{Float8_E4M3FN,2}
const Float8x4_E4M3FN = NVector{Float8_E4M3FN,4}

const Float8x2_E5M2 = NVector{Float8_E5M2,2}
const Float8x4_E5M2 = NVector{Float8_E5M2,4}

const Float8x2_E8M0FNU = NVector{Float8_E8M0FNU,2}
const Float8x4_E8M0FNU = NVector{Float8_E8M0FNU,4}

# 6-bit

const Float6x2_E2M3FN = NVector{Float6_E2M3FN,2}
const Float6x2_E3M2FN = NVector{Float6_E3M2FN,2}

const Float6x4_E2M3FN = NVector{Float6_E2M3FN,4}
const Float6x4_E3M2FN = NVector{Float6_E3M2FN,4}

# 4-bit

const Float4x2_E2M1FN = NVector{Float4_E2M1FN,2}
const Float4x4_E2M1FN = NVector{Float4_E2M1FN,4}
