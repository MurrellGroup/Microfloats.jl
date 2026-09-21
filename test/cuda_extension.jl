using Test
using Microfloats
using Microfloats:
    Float8x2_E4M3FN, Float8x4_E4M3FN,
    Float8x2_E5M2, Float8x4_E5M2,
    Float8x2_E8M0FNU, Float8x4_E8M0FNU,
    Float6x2_E2M3FN, Float6x4_E2M3FN,
    Float6x2_E3M2FN, Float6x4_E3M2FN,
    Float4x2_E2M1FN, Float4x4_E2M1FN

using CUDACore
using CUDACore: CuArray, @cuda
using Microfloats: cvt, bitwidth

samebits(xs, ys) = reinterpret.(UInt8, xs) == reinterpret.(UInt8, ys)
sametuples(xs, ys) = Tuple.(xs) == Tuple.(ys)

gpu_broadcast(::Type{T}, xs) where T = Array(T.(CuArray(xs)))

# A saturating funnel call as an isbits callable, so kernels can take it.
struct NativeCvt{V,M} end
@inline (::NativeCvt{V,M})(x) where {V,M} = cvt(V, x, M(), Microfloats.SAT)

function native_kernel!(out, xs, f)
    i = (CUDACore.blockIdx().x - Int32(1)) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if i <= length(xs)
        @inbounds out[i] = f(xs[i])
    end
    nothing
end

struct GPUConvertWithMode{T,M,O} end
@inline (::GPUConvertWithMode{T,M,O})(x) where {T,M,O} =
    T(x, M(); overflow=O())

gpu_broadcast(::Type{T}, xs, mode::M;
              overflow = Microfloats.overflow_policy(T)) where {T,M} =
    Array(GPUConvertWithMode{T,M,typeof(overflow)}().(CuArray(xs)))

# Vector constructor with a saturating policy; on sm_89+ the fp8 targets
# lower to native cvt.rn.satfinite instructions.
struct GPUVecSAT{V} end
@inline (::GPUVecSAT{V})(xs) where V = V(xs; overflow=Microfloats.SAT)

# NaN payloads of native conversions are hardware-defined; compare NaN lanes
# NaN-aware, everything else bit-exact.
naneq(a, b) = (isnan(a) && isnan(b)) || a === b
naneq_tuples(xs, ys) = all(map((x, y) -> all(naneq.(Tuple(x), Tuple(y))), xs, ys))

const SCALAR_TARGETS = (
    Float8_E5M2, Float8_E4M3, Float8_E3M4, Float8_E4M3FN, Float8_E8M0FNU,
    Float6_E2M3FN, Float6_E3M2FN, Float4_E2M1FN,
)

lane(::Type{T}, x) where T = T(x)
lane(::Type{Microfloats.BFloat16}, x) = Microfloats.BFloat16(x)

svector2(::Type{T}, a, b) where T =
    Microfloats.SVector{2,T}(lane(T, a), lane(T, b))
svector4(::Type{T}, a, b, c, d) where T =
    Microfloats.SVector{4,T}(lane(T, a), lane(T, b), lane(T, c), lane(T, d))

@testset "CUDACore extension" begin
    if CUDACore.functional()
        values = Float32[0, 0.5, 1, 1.5, 2, 3]
        f16s = Float16.(values)
        bf16s = Microfloats.BFloat16.(values)
        f32s = values
        f64s = Float64.(values)
        e4m3s = Float8_E4M3.(values)

        @testset "scalar broadcast" begin
            for xs in (f16s, bf16s, f32s, f64s, e4m3s), T in SCALAR_TARGETS
                @test samebits(gpu_broadcast(T, xs), T.(xs))
            end

            for T in SCALAR_TARGETS
                @test samebits(gpu_broadcast(T, f32s, RoundToZero),
                               T.(f32s, Ref(RoundToZero)))
            end
        end

        f16x2 = [svector2(Float16, 0f0, 1f0), svector2(Float16, 1.5f0, 2f0)]
        bf16x2 = [svector2(Microfloats.BFloat16, 0f0, 1f0), svector2(Microfloats.BFloat16, 1.5f0, 2f0)]
        f32x2 = [svector2(Float32, 0f0, 1f0), svector2(Float32, 1.5f0, 2f0)]
        f64x2 = [svector2(Float64, 0f0, 1f0), svector2(Float64, 1.5f0, 2f0)]
        f4x2 = Float4x2_E2M1FN.(f32x2)

        # Dense fp6 storage packs the unpacked conversion on the device too.
        @testset "dense fp6 broadcast" begin
            for T in (Float6_E2M3FN, Float6_E3M2FN)
                @test sametuples(gpu_broadcast(Microfloats.NVector{T,2}, f32x2),
                                 Microfloats.NVector{T,2}.(f32x2))
            end
        end

        @testset "x2 broadcast" begin
            @test sametuples(gpu_broadcast(Float8x2_E4M3FN, f16x2), Float8x2_E4M3FN.(f16x2))
            @test sametuples(gpu_broadcast(Float8x2_E5M2, bf16x2), Float8x2_E5M2.(bf16x2))
            @test sametuples(gpu_broadcast(Float8x2_E8M0FNU, f64x2), Float8x2_E8M0FNU.(f64x2))
            @test sametuples(gpu_broadcast(Float6x2_E2M3FN, f32x2), Float6x2_E2M3FN.(f32x2))
            @test sametuples(gpu_broadcast(Float6x2_E3M2FN, f4x2), Float6x2_E3M2FN.(f4x2))
            @test sametuples(gpu_broadcast(Float4x2_E2M1FN, f32x2), Float4x2_E2M1FN.(f32x2))
        end

        f16x4 = [
            svector4(Float16, 0f0, 1f0, 1.5f0, 2f0),
            svector4(Float16, 2f0, 3f0, 4f0, 6f0),
        ]
        bf16x4 = [
            svector4(Microfloats.BFloat16, 0f0, 1f0, 1.5f0, 2f0),
            svector4(Microfloats.BFloat16, 2f0, 3f0, 4f0, 6f0),
        ]
        f32x4 = [
            svector4(Float32, 0f0, 1f0, 1.5f0, 2f0),
            svector4(Float32, 2f0, 3f0, 4f0, 6f0),
        ]
        f64x4 = [
            svector4(Float64, 0f0, 1f0, 1.5f0, 2f0),
            svector4(Float64, 2f0, 3f0, 4f0, 6f0),
        ]
        f4x4 = Float4x4_E2M1FN.(f32x4)

        @testset "x4 broadcast" begin
            @test sametuples(gpu_broadcast(Float8x4_E4M3FN, f16x4), Float8x4_E4M3FN.(f16x4))
            @test sametuples(gpu_broadcast(Float8x4_E5M2, bf16x4), Float8x4_E5M2.(bf16x4))
            @test sametuples(gpu_broadcast(Float8x4_E8M0FNU, f64x4), Float8x4_E8M0FNU.(f64x4))
            @test sametuples(gpu_broadcast(Float6x4_E2M3FN, f32x4), Float6x4_E2M3FN.(f32x4))
            @test sametuples(gpu_broadcast(Float6x4_E3M2FN, f4x4), Float6x4_E3M2FN.(f4x4))
            @test sametuples(gpu_broadcast(Float4x4_E2M1FN, f32x4), Float4x4_E2M1FN.(f32x4))
        end

        # Saturating policy: exercises the native `.satfinite` overrides on
        # capable hardware (fp8 on sm_89+, fp6/fp4/ue8m0 on sm_100a) and the
        # generic fallback elsewhere — device must match host either way.
        sat_scalar = Float32[0, -0.0, 0.5, 1.5, -2, 447, 448, 449, -1e9,
                             57344, 6e4, 1e9, Inf, -Inf, NaN]
        @testset "saturating scalar broadcast" begin
            for T in (Float8_E4M3FN, Float8_E5M2)
                got = gpu_broadcast(T, sat_scalar, RoundNearest; overflow=Microfloats.SAT)
                want = T.(sat_scalar, Ref(RoundNearest); overflow=Microfloats.SAT)
                @test all(naneq.(got, want))
            end
            e8m0_vals = Float32[0.5, 1, 3, 2f0^-127, 1e30, Inf, NaN]
            got = gpu_broadcast(Float8_E8M0FNU, e8m0_vals, RoundToZero; overflow=Microfloats.SAT)
            want = Float8_E8M0FNU.(e8m0_vals, Ref(RoundToZero); overflow=Microfloats.SAT)
            @test all(naneq.(got, want))
        end

        sat2 = [svector2(Float32, 448f0, 1f9), svector2(Float32, -1f9, 0.5f0),
                svector2(Float32, NaN32, 2f0), svector2(Float32, 6f4, -6f4)]
        sat4 = [svector4(Float32, 448f0, 1f9, -1f9, 0.5f0),
                svector4(Float32, NaN32, 2f0, 6f4, -6f4)]
        @testset "saturating x2/x4 broadcast" begin
            for V in (Float8x2_E4M3FN, Float8x2_E5M2)
                @test naneq_tuples(Array(GPUVecSAT{V}().(CuArray(sat2))), GPUVecSAT{V}().(sat2))
            end
            for V in (Float8x4_E4M3FN, Float8x4_E5M2)
                @test naneq_tuples(Array(GPUVecSAT{V}().(CuArray(sat4))), GPUVecSAT{V}().(sat4))
            end
        end

        # The fp6/fp4/ue8m0 instructions exist only on arch-specific targets, so
        # a default (baseline) compile never reaches them. Compile explicitly for
        # the device's `a` target and compare every lane with the host funnel.
        cap = CUDACore.capability(CUDACore.device())
        if cap >= v"10.0"
            @testset "native conversions on sm_$(cap.major)$(cap.minor)a" begin
                arch = CUDACore.SMVersion(cap.major, cap.minor, :arch)
                native(f, xs) = begin
                    ys = CuArray(xs)
                    out = similar(ys, Core.Compiler.return_type(f, Tuple{eltype(xs)}))
                    kernel = @cuda launch=false arch=arch native_kernel!(out, ys, f)
                    kernel(out, ys, f; threads=256, blocks=cld(length(xs), 256))
                    Array(out)
                end
                bits(x) = reinterpret(UInt8, [x...])

                # Every representable value, the midpoints between neighbours
                # (ties), their float neighbours, and out-of-range magnitudes.
                function probes(::Type{T}) where T
                    vals = sort!(unique!(filter(isfinite, Float32.(reinterpret.(T, 0x00:UInt8(2^bitwidth(T) - 1))))))
                    mids = (vals[1:end-1] .+ vals[2:end]) ./ 2
                    edge = Float32[floatmax(T) * 1.01f0, floatmax(T) * 4, 1f30, Inf32, -0.0f0]
                    xs = vcat(vals, mids, nextfloat.(mids), prevfloat.(mids), edge, -edge)
                    T === Float8_E8M0FNU ? filter(x -> !signbit(x), xs) : xs
                end

                for (T, mode) in ((Float6_E2M3FN, RoundNearest), (Float6_E3M2FN, RoundNearest),
                                  (Float4_E2M1FN, RoundNearest), (Float8_E8M0FNU, RoundToZero))
                    xs = probes(T)
                    ys = reverse(xs)
                    pairs = [(a, b) for (a, b) in zip(xs, ys)]
                    quads = [(a, b, b, a) for (a, b) in zip(xs, ys)]
                    V2 = T === Float4_E2M1FN || T === Float8_E8M0FNU ? Microfloats.NVector{T,2} : Microfloats.SVector{2,T}
                    V4 = T === Float4_E2M1FN || T === Float8_E8M0FNU ? Microfloats.NVector{T,4} : Microfloats.SVector{4,T}
                    M = typeof(mode)
                    host(V, ps) = map(NativeCvt{V,M}(), ps)
                    # UE8M0 kernels do not compile on the device yet: the module
                    # keeps unused Julia string-runtime lookups, the same failure
                    # as the UE8M0 broadcasts above.
                    agree(f) = T === Float8_E8M0FNU ? (@test_broken f()) : (@test f())
                    agree(() -> bits(native(NativeCvt{T,M}(), xs)) == bits(host(T, xs)))
                    agree(() -> native(NativeCvt{V2,M}(), pairs) == host(V2, pairs))
                    agree(() -> native(NativeCvt{V4,M}(), quads) == host(V4, quads))
                    if V2 <: Microfloats.SVector
                        # dense destinations: native unpacked result, then packed
                        N2, N4 = Microfloats.NVector{T,2}, Microfloats.NVector{T,4}
                        @test native(NativeCvt{N2,M}(), pairs) == host(N2, pairs)
                        @test native(NativeCvt{N4,M}(), quads) == host(N4, quads)
                    end
                end
            end
        end
    else
        @test_skip "CUDACore.functional() == false"
    end
end
