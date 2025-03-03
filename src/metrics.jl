using Interpolations: linear_interpolation
using Statistics: cor
using LinearAlgebra: diag
using OneHotArrays: OneHotMatrix

include("network.jl")

function arc_error(phase::Real)
    return sin(pi * phase)
end

function arc_error(phases::AbstractArray)
    return arc_error.(phases)
end

function quadrature_loss(phases::AbstractArray, truth::AbstractArray)
    #truth = 2.0 .* truth .- 1.0
    targets = 0.5 .* truth
    sim = similarity(phases, targets, dim = 1)
    return 1.0 .- sim
end

function similarity_loss(phases::AbstractArray, truth::AbstractArray, dim::Int)
    sim = similarity(phases, truth, dim = dim)
    return 1.0 .- sim
end

function z_score(phases::AbstractArray)
    arc = remap_phase(phases .- 0.5)
    score = abs.(atanh.(arc))
    return score
end

function loss_and_accuracy(data_loader, model, ps, st, args)
    if args.use_cuda
        dev = gdev
    else
        dev = cdev
    end

    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x = x |> dev
        
        ŷ, _ = model(x, ps, st)

        if typeof(ŷ) <: SpikingTypes
            ŷ, _ = train_to_phase(ŷ)
        end
        
        ls += sum(quadrature_loss(ŷ, 1.0 .* y |> dev))
        acc += sum(accuracy_quadrature(ŷ, y)) ## Decode the output of the model
        num +=  size(y)[2]
    end
    return ls/num, acc/num
end

function dense_onehot(x::OneHotMatrix)
    return 1.0f0 .* x
end

function spiking_accuracy(data_loader, model, ps, st, args, repeats::Int)
    acc = []
    n_phases = []
    num = 0

    for (x, y) in data_loader
        if args.use_cuda
            x = x |> gdev
            y = y |> dense_onehot |> gdev
        end
        
        spk_output, _ = model(x, ps, st)
        ŷ = train_to_phase(spk_output)
        
        append!(acc, sum.(accuracy_quadrature(ŷ, y))) ## Decode the output of the model
        num += size(x)[end]
    end

    acc = sum(reshape(acc, repeats, :), dims=2) ./ num
    return acc
end

function predict_quadrature(phases::AbstractMatrix)
    if on_gpu(phases)
        phases = phases |> cdev
    end

    predictions = getindex.(argmin(abs.(phases .- 0.5), dims=1), 1)'
    return predictions
end

function predict_quadrature(spikes::SpikingCall)
    phases = train_to_phase(spikes)[end-1, :, :]
    return predict_quadrature(phases)
end

function accuracy_quadrature(phases::AbstractMatrix, truth::AbstractMatrix)
    if on_gpu(phases, truth)
        phases = phases |> cdev
        truth = truth |> cdev
    end

    predictions = predict_quadrature(phases)
    labels = getindex.(findall(truth .== 1.0f0), 1)
    return predictions .== labels
end

function accuracy_quadrature(phases::Array{<:Real,3}, truth::AbstractMatrix)
    if on_gpu(phases, truth)
        phases = phases |> cdev
        truth = truth |> cdev
    end

    return [accuracy_quadrature(phases[i,:,:], truth) for i in axes(phases,1)]
end

function confusion_matrix(sim, truth, threshold::Real)
    truth = hcat(truth .== 1, truth .== 0)
    prediction = hcat(sim .> threshold, sim .<= threshold)

    confusion = truth' * prediction
    return confusion
end

function cycle_correlation(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3})
    n_cycles = axes(dynamic_phases, 1)
    cor_vals = [cor_realvals(static_phases |> vec, dynamic_phases[i,:,:] |> vec) for i in n_cycles]
    return cor_vals
end

function cor_realvals(x, y)
    is_real = x -> .!isnan.(x)
    x_real = is_real(x)
    y_real = is_real(y)
    reals = x_real .* y_real
    if sum(reals) == 0
        return 0.0
    else
        return cor(x[reals], y[reals])
    end
end

function OvR_matrices(predictions, labels, threshold::Real)
    #get the confusion matrix for each class verus the rest
    mats = diag([confusion_matrix(ys, ts, threshold) for ys in eachslice(predictions, dims=1), ts in eachslice(labels, dims=1)])
    return mats
end

function tpr_fpr(prediction, labels, points::Int = 201, epsilon::Real = 0.01)
    test_points = range(start = 0.0, stop = -20.0, length = points)
    test_points = vcat(exp.(test_points), 0.0, reverse(-1 .* exp.(test_points)))

    fn = x -> sum(OvR_matrices(prediction, labels, x))
    confusion = cat(fn.(test_points)..., dims=3)

    classifications = dropdims(sum(confusion, dims=2), dims=2)
    cond_true = classifications[1,:]
    cond_false = classifications[2,:]

    #return cond_true, cond_false

    true_positives = confusion[1,1,:]
    false_positives = confusion[2,1,:]

    #return true_positives, false_positives

    tpr = true_positives ./ cond_true
    fpr = false_positives ./ cond_false

    return tpr, fpr
end
    
function interpolate_roc(roc)
    tpr, fpr = roc
    interp = linear_interpolation(fpr, tpr)
    return interp
end