# This file is intended to be included and run from runtests.jl
# Ensure that runtests.jl has already loaded PhasorNetworks and other common dependencies.

"""
Run all the basic VSA tests and check they pass
"""
function vsa_tests()
    @testset "VSA Tests" begin
        @info "Running VSA tests..."

        test_orthogonal()
        test_outer()
        test_binding()
        test_bundling()
        test_similarity_outer_complex_cpu_shape()
        test_similarity_outer_rrule()
        test_angle_to_complex_rrule()
    end
end

"""
Check if a value is within the bounds determined by epsilon
"""
function in_tolerance(x)
    return abs(x) < epsilon ? true : false
end

"""
Produce all possible pairs of angles
"""
function make_angle_pairs()
    phases = collect([[x, y] for x in range(-1.0, 1.0, n_x), y in range(-1.0, 1.0, n_y)]) |> stack
    phases = reshape(phases, (1,2,:))
    return phases
end

"""
Remove NaN values
"""
function remove_nan(x)
    y = filter(x -> !isnan(x), x)
    return y
end

"""
Produce the error of angular values from zero
"""
function sin_error(x)
    return sin.(pi .* x) |> mean
end

"""
Basic test that random VSA symbols are orthogonal
"""
function test_orthogonal()
    @testset "Orthogonality Test" begin
        x = random_symbols((100, 1024))
        y = random_symbols((100, 1024))

        s = similarity(x, y)
        @test mean(s) < 0.1 ? true : false
    end
end

"""
Test the outer similarity function with normal and spiking arguments
"""
function test_outer()
    @testset "Outer Similarity Test" begin
        function check_phase(matrix)
            in_phase = diag(matrix)
            anti_phase = diag(matrix, convert(Int, round(n_x / 2)))

            v1 = reduce(*, map(x -> x > 1.0 - epsilon, in_phase))
            v2 = reduce(*, map(x -> x < -1.0 + epsilon, anti_phase))
            return v1, v2
        end

        #check the non-spiking implementation
        phase_x = reshape(range(-1.0, 1.0, n_x), (n_vsa, n_x, 1)) |> collect
        phase_y = reshape(range(-1.0, 1.0, n_y), (n_vsa, n_y, 1)) |> collect
        sims = similarity_outer(phase_x, phase_y, dims=2)[:,:,1]
        v1, v2 = check_phase(sims)
        @test v1 && v2

        #check the spiking implementation
        st_x = phase_to_train(phase_x, spk_args = spk_args, repeats = repeats)
        st_y = phase_to_train(phase_y, spk_args = spk_args, repeats = repeats)
        sims_2 = similarity_outer(st_x, st_y, tspan=tspan, spk_args = spk_args)
        #check at the last time step
        sims_spk = sims_2[end][:,:,1]
        v1s, v2s = check_phase(sims_spk)
        @test v1s && v2s

        #check the cross-implementation error
        avg_error = mean(sims .- sims_spk)
        @test avg_error < epsilon
    end
end


"""
Test the binding/unbinding functions in both versions & cross-check errors
"""
function test_binding()
    @testset "Binding Test" begin
        phases = make_angle_pairs()
        #check binding and unbinding functions
        b = v_bind(phases, dims=2)
        ub = v_unbind(phases[1:1,1:1,:], phases[1:1,2:2,:])

        #check binding via oscillators
        st_x = phase_to_train(phases[1:1,1:1,:], spk_args=spk_args, repeats = repeats)
        st_y = phase_to_train(phases[1:1,2:2,:], spk_args=spk_args, repeats = repeats)
        soln = v_bind(st_x, st_y, spk_args=spk_args, tspan=tspan, return_solution=true);
        decoded = solution_to_phase(soln, tbase, spk_args=spk_args);
        u_err = mean(decoded[1,:,:,:] .- b[1,:,:], dims=(1,2))[end]
        u_check = in_tolerance(u_err)
        @test u_check

        #check with spiking outputs
        b2 = v_bind(st_x, st_y, spk_args=spk_args, tspan=tspan, return_solution=false)
        b2d = train_to_phase(b2, spk_args=spk_args)
        enc_error = remove_nan(vec(b2d[:,:,:,5]) .- vec(b)) |> mean
        enc_check = in_tolerance(enc_error)
        @test enc_check 

        #check unbinding operation
        ub_soln = v_unbind(st_x, st_y, tspan=tspan, spk_args=spk_args, return_solution=true)
        decoded = solution_to_phase(ub_soln, tbase, spk_args=spk_args)
        err = mean(decoded[1,:,:,:] .- ub[1,:,:], dims=(1,2))[end]
        unbind_chk = in_tolerance(err)
        @test unbind_chk 

        #check unbinding with spiking outputs
        ub2 = v_unbind(st_x, st_y, spk_args=spk_args, tspan=tspan, return_solution=false)
        ub2d = train_to_phase(ub2, spk_args=spk_args)
        ub_enc_error = remove_nan(vec(ub2d[:,:,:,5]) .- vec(ub)) |> mean
        ub_enc_check = in_tolerance(ub_enc_error)
        @test ub_enc_check 
    end
end

"""
Central-difference Jacobian for a complex array argument, ChainRules
convention (real-valued cost L; tangent stored as ∂L/∂real + i·∂L/∂imag).
"""
function _fd_complex_grad(f, A::AbstractArray{ComplexF32}; ε::Float32=1f-3)
    g = zeros(ComplexF32, size(A))
    for I in eachindex(A)
        z0 = A[I]
        A[I] = z0 + ε;     fpr = f(A)
        A[I] = z0 - ε;     fmr = f(A)
        A[I] = z0 + ε*im;  fpi = f(A)
        A[I] = z0 - ε*im;  fmi = f(A)
        A[I] = z0
        g[I] = (fpr - fmr) / (2ε) + im * (fpi - fmi) / (2ε)
    end
    return g
end

"""
Lock in the canonical-shape semantics of the CPU complex dispatch
`similarity_outer(::AbstractArray{<:Complex,3}, ...)`. Returns
`(M, N, X)` and agrees with the helper. The previous comprehension-based
implementation averaged over the batch dim and returned `(M, N, D)`.
"""
function test_similarity_outer_complex_cpu_shape()
    @testset "similarity_outer CPU complex 3D shape" begin
        rng = Xoshiro(11)
        D, M, N, X = 4, 5, 6, 3
        A = ComplexF32.(randn(rng, ComplexF64, D, M, X))
        B = ComplexF32.(randn(rng, ComplexF64, D, N, X))

        out = similarity_outer(A, B; dims=2)
        @test size(out) == (M, N, X)
        @test out ≈ PhasorNetworks._similarity_outer_canonical_complex(A, B)

        # 2D variant matches the CPU real 2-D convention (transposed).
        A2 = ComplexF32.(randn(rng, ComplexF64, D, M))
        B2 = ComplexF32.(randn(rng, ComplexF64, D, N))
        out2 = similarity_outer(A2, B2; dims=2)
        @test size(out2) == (N, M)
    end
end

"""
Verify the closed-form rrule for `_similarity_outer_canonical_complex`
against finite-difference gradients and against the broadcast-traced
reference (Zygote on the forward kernel). Runs on CPU.
"""
function test_similarity_outer_rrule()
    @testset "similarity_outer rrule" begin
        rng = Xoshiro(0)
        D, M, N, X = 3, 4, 5, 2
        A = ComplexF32.(randn(rng, ComplexF64, D, M, X))
        B = ComplexF32.(randn(rng, ComplexF64, D, N, X))

        # Forward agrees with a naive scalar reference.
        out = PhasorNetworks._similarity_outer_canonical_complex(A, B)
        ref = zeros(Float32, M, N, X)
        invD = inv(Float32(D))
        for x in 1:X, n in 1:N, m in 1:M, d in 1:D
            ref[m, n, x] += invD * (0.5f0 * abs2(A[d,m,x] + B[d,n,x]) - 1f0)
        end
        @test maximum(abs.(out .- ref)) < 1f-5

        # Pullback w.r.t. a random output cotangent.
        ḡ = Float32.(randn(rng, M, N, X))
        _, pb = ChainRulesCore.rrule(
            PhasorNetworks._similarity_outer_canonical_complex, A, B)
        _, dA, dB = pb(ḡ)

        # Finite-difference reference.
        fA = A_ -> sum(ḡ .* PhasorNetworks._similarity_outer_canonical_complex(A_, B))
        fB = B_ -> sum(ḡ .* PhasorNetworks._similarity_outer_canonical_complex(A, B_))
        fdA = _fd_complex_grad(fA, copy(A))
        fdB = _fd_complex_grad(fB, copy(B))

        @test maximum(abs.(dA .- fdA)) < 5f-3
        @test maximum(abs.(dB .- fdB)) < 5f-3

        # Zygote.gradient(...) routes through our rrule; should match the
        # direct pullback bit-for-bit.
        gA, gB = Zygote.gradient(
            (a, b) -> sum(ḡ .* PhasorNetworks._similarity_outer_canonical_complex(a, b)),
            A, B)
        @test maximum(abs.(gA .- dA)) < 1f-5
        @test maximum(abs.(gB .- dB)) < 1f-5
    end
end

function test_angle_to_complex_rrule()
    @testset "angle_to_complex rrule" begin
        rng = Xoshiro(0)
        x = Float32.(2 .* rand(rng, Float32, 3, 4) .- 1)  # phases in [-1,1]
        z = angle_to_complex(x)

        # Random complex cotangent for output.
        dz = ComplexF32.(randn(rng, ComplexF64, size(z)))

        _, pb = ChainRulesCore.rrule(angle_to_complex, x)
        _, dx = pb(dz)

        # Finite-difference reference: real input, real-valued contraction
        # against complex cotangent uses the same `real(conj(dz)·δz)`
        # convention, which for real δx reduces to summing real parts.
        ε = 1f-3
        fdx = zeros(Float32, size(x))
        f = x_ -> sum(real.(conj.(dz) .* angle_to_complex(x_)))
        for I in eachindex(x)
            x0 = x[I]
            x[I] = x0 + ε; fp = f(x)
            x[I] = x0 - ε; fm = f(x)
            x[I] = x0
            fdx[I] = (fp - fm) / (2ε)
        end
        @test maximum(abs.(dx .- fdx)) < 5f-3
    end
end

function test_bundling()
    @testset "Bundling test" begin
        phases = make_angle_pairs()
        #check bundling function
        b = v_bundle(phases, dims=2);

        st = phase_to_train(phases, spk_args=spk_args, repeats=6)
        #check potential encodings
        b2_sol = v_bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=true)
        b2_phase = solution_to_phase(b2_sol, tbase, spk_args=spk_args, offset=0.0)
        b2_phase_error = vec(b2_phase[1,1,:,end]) .- vec(b) |> mean
        b2_phase_check = in_tolerance(b2_phase_error)
        @test b2_phase_check 

        #check spiking encoding
        b2 = v_bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=false)
        decoded = train_to_phase(b2, spk_args=spk_args)
        b2_error = remove_nan(vec(decoded[:,:,:,end-1]) .- vec(b)) |> sin_error
        b2_spike_check = in_tolerance(b2_error)
        @test b2_spike_check
    end
end