include("network.jl")

function arc_error(phase::Real)
    return sin(pi_f32 * phase)
end

function arc_error(phases::AbstractArray)
    return arc_error.(phases)
end

function angular_mean(phases::AbstractArray; dims)
    u = exp.(pi_f32 * 1.0f0im .* phases)
    u_mean = mean(u, dims=dims)
    phase = angle.(u_mean) ./ pi_f32
    return phase
end

function exp_score(similarity::AbstractArray; scale::Real = 3.0f0)
    return exp.((1.0f0 .- similarity) .* scale) .- 1.0f0
end

function quadrature_loss(phases::AbstractArray, truth::AbstractArray)
    targets = 0.5f0 .* truth
    sim = similarity(phases, targets, dim = 1)
    return 1.0f0 .- sim
end

function codebook_loss(similarities::AbstractArray, truth::AbstractArray)
    distance = abs.(1.0 .- similarities) .* truth
    distance = sum(distance .* truth, dims=1)
    loss = 2.0f0 .* sin.(pi_f32/4.0f0 .* distance) .^ 2.0f0
    return loss
end
function similarity_loss(phases::AbstractArray, truth::AbstractArray; dim::Int = 1)
    sim = similarity(phases, truth, dim = dim)
    return 1.0f0 .- sim
end

function z_score(phases::AbstractArray)
    arc = remap_phase(phases .- 0.5f0)
    score = abs.(atanh.(arc))
    return score
end

function loss_and_accuracy(data_loader, model, ps, st, args; loss_fn = codebook_loss, predict_fn = predict_codebook)
    if args.use_cuda && CUDA.functional()
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
        
        ls += sum(loss_fn(ŷ, 1.0f0 .* y |> dev))
        acc += sum(eval_accuracy(ŷ, y, predict_fn=predict_fn)) ## Decode the output of the model
        num +=  size(y)[2]
    end
    return ls/num, acc/num
end

function evaluate_codebook(data_loader, model, ps, st, args)
    return loss_and_accuracy(data_loader, model, ps, st, args, loss_fn=codebook_loss, predict_fn=predict_codebook)
end

function evaluate_quadrature(data_loader, model, ps, st, args)
    return loss_and_accuracy(data_loader, model, ps, st, args, loss_fn=quadrature_loss, predict_fn=predict_quadrature)
end

function dense_onehot(x::OneHotMatrix)
    return 1.0f0 .* x
end

function spiking_accuracy(data_loader, model, ps, st, args)
    acc = []
    n_phases = []
    num = 0

    n_batches = length(data_loader)

    for (x, y) in data_loader
        if args.use_cuda && CUDA.functional()
            x = x |> gdev
            y = y |> dense_onehot |> gdev
        end
        
        spk_output, _ = model(x, ps, st)
        ŷ = train_to_phase(spk_output)
        
        append!(acc, sum.(accuracy_quadrature(ŷ, y))) ## Decode the output of the model
        num += size(x)[end]
    end

    acc = sum(reshape(acc, :, n_batches), dims=2) ./ num
    return acc
end

function predict_quadrature(phases::AbstractMatrix)
    if on_gpu(phases)
        phases = phases |> cdev
    end

    predictions = getindex.(argmin(abs.(phases .- 0.5f0), dims=1), 1)'
    return predictions
end

function predict_quadrature(spikes::SpikingCall)
    phases = train_to_phase(spikes)[end-1, :, :]
    return predict_quadrature(phases)
end

function predict_codebook(sims::AbstractMatrix; dims=1)
    if dims == -1
        dims = ndims(sims)
    end

    predictions = vec(getindex.(argmax(sims, dims=dims), dims))
    return predictions
end

function eval_accuracy(phases::AbstractMatrix, truth::AbstractMatrix; predict_fn::Function = predict_codebook)
    if on_gpu(phases, truth)
        phases = phases |> cdev
        truth = truth |> cdev
    end

    predictions = predict_fn(phases)
    labels = getindex.(findall(truth .== 1.0f0), 1)
    return predictions .== labels
end

function eval_accuracy(phases::Array{<:Real,3}, truth::AbstractMatrix; predict_fn::Function = predict_codebook)
    if on_gpu(phases, truth)
        phases = phases |> cdev
        truth = truth |> cdev
    end

    return [eval_accuracy(phases[i,:,:], truth, predict_fn=predict_fn) for i in axes(phases,1)]
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

function cycle_sparsity(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3})
    n_cycles = axes(dynamic_phases, 1)
    total = reduce(*, size(static_phases))
    sparsity_vals = [sum(isnan.(dynamic_phases[i,:,:])) / total for i in n_cycles]
    return sparsity_vals
end

function cor_realvals(x, y)
    is_real = x -> .!isnan.(x)
    x_real = is_real(x)
    y_real = is_real(y)
    reals = x_real .* y_real
    if sum(reals) == 0
        return 0.0f0
    else
        return cor(x[reals], y[reals])
    end
end

function OvR_matrices(predictions, labels, threshold::Real)
    #get the confusion matrix for each class verus the rest
    mats = diag([confusion_matrix(ys, ts, threshold) for ys in eachslice(predictions, dims=1), ts in eachslice(labels, dims=1)])
    return mats
end

function tpr_fpr(prediction, labels, points::Int = 201, epsilon::Real = 0.01f0)
    test_points = range(start = 0.0f0, stop = -20.0f0, length = points)
    test_points = vcat(exp.(test_points), 0.0f0, reverse(-1.0f0 .* exp.(test_points)))
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