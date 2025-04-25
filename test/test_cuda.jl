using Pkg
Pkg.activate(".")

include("../src/PhasorNetworks.jl")
using .PhasorNetworks, Plots, DifferentialEquations

n_x = 101
n_y = 101
n_vsa = 1
repeats = 6
tspan = (0.0, repeats*1.0)

function bundling_test(spk_args::SpikingArgs)
    tbase = collect(0.0:0.01:tspan[2])
    phases = collect([[x, y] for x in range(-1.0, 1.0, n_x), y in range(-1.0, 1.0, n_y)]) |> stack
    phases = reshape(phases, (1,2,:))
    st = phase_to_train(phases, spk_args, repeats=6)
    #check potential encodings
    b2_sol = v_bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=true)
    b2_phase = solution_to_phase(b2_sol, tbase, spk_args=spk_args, offset=0.0)
    b2_phase_error = vec(b2_phase[1,1,:,end]) .- vec(b)
    
    return b2_sol, b2_phase_error
end

tbase = collect(0.0:0.01:tspan[2]);
phases = collect([[x, y] for x in range(-1.0, 1.0, n_x), y in range(-1.0, 1.0, n_y)]) |> stack
phases = reshape(phases, (1,2,:));
b = v_bundle(phases, dims=2);

spk_args = SpikingArgs(solver = Heun(),
                    solver_args = Dict(:adaptive => false, 
                                    :dt => 0.01),
                                    threshold = 0.001)

st = phase_to_train(phases, spk_args=spk_args, repeats=6)
stg = SpikeTrainGPU(st)
b2_sol = v_bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=true)
b2_solg = v_bundle(stg, dims=2, spk_args=spk_args, tspan=tspan, return_solution=true)
print(b2_solg)