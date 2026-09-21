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

# The widening funnel as an isbits callable.
struct NativeWiden{V} end
@inline (::NativeWiden{V})(x) where V = cvt(V, x)

# PTX assembly of a one-element kernel applying `f`, to check which
# instructions a conversion lowers to.
function asm_kernel!(out, xs, f)
    @inbounds out[1] = f(xs[1])
    nothing
end
function device_asm(f, ::Type{X}, arch) where X
    device_type(T) = typeof(CUDACore.cudaconvert(CuArray{T}(undef, 1)))
    Y = Core.Compiler.return_type(f, Tuple{X})
    config = CUDACore.compiler_config(CUDACore.device(); kernel=true, arch=arch)
    GPUCompiler = CUDACore.GPUCompiler
    mi = GPUCompiler.methodinstance(typeof(asm_kernel!), Tuple{device_type(Y),device_type(X),typeof(f)})
    GPUCompiler.JuliaContext() do _
        GPUCompiler.compile(:asm, GPUCompiler.CompilerJob(mi, config))[1]
    end
end
ptxas_has(v) = any(>=(v), CUDACore.ptxas_compat().ptx)

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
    Float8_E5M2, Float8_E4M3, Float8_E3M4, Float8_E4M3FN, Float8_E8M0FNU, Float8_E5M3FNU,
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

        # Native conversions against the host funnel, compiled explicitly for the
        # device's arch- and family-specific targets (a baseline target has
        # only the fp8 forms). The lane sweeps cover every representable
        # value, every tie between neighbours, their float neighbours and the
        # saturating range, from Float32, Float16 and BFloat16 sources; the
        # widening sweeps cover every bit pattern.
        cap = CUDACore.capability(CUDACore.device())
        feature_sets = cap >= v"10.0" ? (:arch, :family) : (:baseline,)
        @testset "native conversions on $(CUDACore.SMVersion(cap.major, cap.minor, fs))" for fs in feature_sets
            arch = CUDACore.SMVersion(cap.major, cap.minor, fs)
            mxfp = fs !== :baseline
            function native(f, xs)
                ys = CuArray(xs)
                out = similar(ys, Core.Compiler.return_type(f, Tuple{eltype(xs)}))
                kernel = @cuda launch=false arch=arch native_kernel!(out, ys, f)
                kernel(out, ys, f; threads=256, blocks=cld(length(xs), 256))
                Array(out)
            end
            same(a, b) = all(naneq.(Tuple(a), Tuple(b)))
            agree(got, want) = all(map(same, got, want))

            # Every representable value, the midpoints between neighbours
            # (ties), their float neighbours, and out-of-range magnitudes.
            function probes(::Type{T}, ::Type{S}) where {T,S}
                vals = sort!(unique!(filter(isfinite, Float32.(reinterpret.(T, 0x00:UInt8(2^bitwidth(T) - 1))))))
                mids = S.((vals[1:end-1] .+ vals[2:end]) ./ 2)
                # (including Float32 subnormals, below even E8M0's 2^-127)
                edge = S.(Float32[Float32(floatmax(T)) * 1.01f0, Float32(floatmax(T)) * 4, 1f30, Inf32, -0.0f0,
                                  2f0^-130, 3 * 2f0^-129, nextfloat(0f0)])
                xs = vcat(S.(vals), mids, nextfloat.(mids), prevfloat.(mids), edge, -edge)
                Microfloats.hasnan(T) && push!(xs, S(NaN))
                Microfloats.sign_bits(T) == 0 ? filter(x -> !signbit(x), xs) : xs
            end

            narrowing = (
                (Float8_E4M3FN,  (RoundNearest,),            (Float32, Float16, Microfloats.BFloat16), true),
                (Float8_E5M2,    (RoundNearest,),            (Float32, Float16, Microfloats.BFloat16), true),
                (Float6_E2M3FN,  (RoundNearest,),            (Float32, Float16, Microfloats.BFloat16), mxfp),
                (Float6_E3M2FN,  (RoundNearest,),            (Float32, Float16, Microfloats.BFloat16), mxfp),
                (Float4_E2M1FN,  (RoundNearest,),            (Float32, Float16, Microfloats.BFloat16), mxfp),
                (Float8_E8M0FNU, (RoundToZero, RoundUp),     (Float32, Microfloats.BFloat16),          mxfp),
            )
            for (T, modes, sources, _) in narrowing, mode in modes, S in sources
                xs = probes(T, S)
                ys = reverse(xs)
                pairs = collect(zip(xs, ys))
                quads = [(a, b, b, a) for (a, b) in pairs]
                M = typeof(mode)
                for (V, inputs) in ((T, xs),
                                    (Microfloats.SVector{2,T}, pairs), (Microfloats.SVector{4,T}, quads),
                                    (Microfloats.NVector{T,2}, pairs), (Microfloats.NVector{T,4}, quads))
                    f = NativeCvt{V,M}()
                    @test agree(native(f, inputs), map(f, inputs))
                end
            end

            widening = (
                (Float8_E4M3FN,  (Float16, Microfloats.BFloat16, Float32)),
                (Float8_E5M2,    (Float16, Microfloats.BFloat16, Float32)),
                (Float6_E2M3FN,  (Float16, Microfloats.BFloat16, Float32)),
                (Float6_E3M2FN,  (Float16, Microfloats.BFloat16, Float32)),
                (Float4_E2M1FN,  (Float16, Microfloats.BFloat16, Float32)),
                (Float8_E8M0FNU, (Microfloats.BFloat16, Float32)),
            )
            for (T, destinations) in widening, F in destinations
                xs = reinterpret.(T, 0x00:UInt8(2^bitwidth(T) - 1))
                ys = reverse(xs)
                quads = [Microfloats.SVector{4,T}(a, b, b, a) for (a, b) in zip(xs, ys)]
                packed = [Microfloats.NVector{T,2}((a, b)) for (a, b) in zip(xs, ys)]
                for (V, inputs) in ((F, xs), (Microfloats.SVector{4,F}, quads), (Microfloats.SVector{2,F}, packed))
                    f = NativeWiden{V}()
                    @test agree(native(f, inputs), map(f, inputs))
                end
            end

            # The sweeps pass on the generic path too, so check separately
            # that each form really lowers to its instruction.
            @testset "instruction selection" begin
                # The generic path must also be gone. A gate that does not fold
                # leaves both paths behind a runtime branch: symbol comparisons,
                # the lanewise fallback or the widening lookup table survive.
                function selected(f, X, instr)
                    asm = device_asm(f, X, arch)
                    occursin(instr, asm) && !occursin("jl_sym", asm) &&
                        !occursin("cvt_lanes", asm) && !occursin("_j_const", asm)
                end
                sat(V) = NativeCvt{V,typeof(RoundNearest)}()
                V2(T) = Microfloats.SVector{2,T}
                D2(T) = Microfloats.NVector{T,2}                              # packed destination
                P2(T) = typeof(Microfloats.NVector{T,2}((one(T), one(T))))    # packed source, concrete
                @test selected(sat(V2(Float8_E4M3FN)), NTuple{2,Float32}, "cvt.rn.satfinite.e4m3x2.f32")
                @test selected(sat(V2(Float8_E5M2)), NTuple{2,Float16}, "cvt.rn.satfinite.e5m2x2.f16x2")
                @test selected(NativeWiden{V2(Float16)}(), P2(Float8_E4M3FN), "cvt.rn.f16x2.e4m3x2")
                if mxfp
                    @test selected(sat(V2(Float6_E2M3FN)), NTuple{2,Float32}, "cvt.rn.satfinite.e2m3x2.f32")
                    @test selected(sat(D2(Float6_E3M2FN)), NTuple{2,Float32}, "cvt.rn.satfinite.e3m2x2.f32")
                    @test selected(sat(D2(Float4_E2M1FN)), NTuple{2,Float32}, "cvt.rn.satfinite.e2m1x2.f32")
                    @test selected(NativeCvt{V2(Float8_E8M0FNU),typeof(RoundUp)}(), NTuple{2,Float32},
                                   "cvt.rp.satfinite.ue8m0x2.f32")
                    @test selected(NativeWiden{V2(Float16)}(), P2(Float4_E2M1FN), "cvt.rn.f16x2.e2m1x2")
                    @test selected(NativeWiden{V2(Microfloats.BFloat16)}(), P2(Float8_E8M0FNU), "cvt.rn.bf16x2.ue8m0x2")
                    if ptxas_has(v"9.1")
                        @test selected(sat(D2(Float4_E2M1FN)), NTuple{2,Microfloats.BFloat16},
                                       "cvt.rn.satfinite.e2m1x2.bf16x2")
                    end
                    if ptxas_has(v"9.2")
                        @test selected(NativeWiden{V2(Microfloats.BFloat16)}(), P2(Float4_E2M1FN), "cvt.rn.bf16x2.e2m1x2")
                        @test selected(NativeWiden{V2(Float32)}(), V2(Float8_E4M3FN), "cvt.rn.bf16x2.e4m3x2")
                    end
                end
            end
        end
    else
        @test_skip "CUDACore.functional() == false"
    end
end
