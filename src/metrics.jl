include("network.jl")

#Misc. metrics

"""
    arc_error(phase::Real) -> Real
    arc_error(phases::AbstractArray) -> AbstractArray

Compute the arc error for phase values, mapping phase differences to sinusoidal error.
This provides a continuous, differentiable metric for phase differences.

# Arguments
- `phase`: Phase value in [-1, 1] range
- `phases`: Array of phase values

# Returns
Sine of the phase multiplied by π, producing smooth error measure in [-1, 1].
"""
function arc_error(phase::Real)
    return sin(pi_f32 * phase)
end

function arc_error(phases::AbstractArray)
    return arc_error.(phases)
end

"""
    angular_mean(phases::AbstractArray; dims) -> AbstractArray

Compute the circular mean of phase values along specified dimensions.
Handles the circular nature of phases by converting to complex exponentials.

# Arguments
- `phases`: Array of phase values in [-1, 1] range
- `dims`: Dimensions along which to compute mean

# Implementation
1. Converts phases to complex exponentials (e^(iπθ))
2. Computes arithmetic mean in complex plane
3. Converts back to phase via angle

# Returns
Phase values representing the circular mean along specified dimensions.
"""
function angular_mean(phases::AbstractArray; dims)
    u = exp.(pi_f32 * 1.0f0im .* phases)
    u_mean = mean(u, dims=dims)
    phase = angle.(u_mean) ./ pi_f32
    return phase
end

"""
    exp_score(similarity::AbstractArray; scale::Real = 3.0f0) -> AbstractArray

Convert similarity values to exponential scores, emphasizing differences.
Useful for converting similarity metrics to loss-like values.

# Arguments
- `similarity`: Array of similarity values in [0, 1] range
- `scale`: Scaling factor for exponential transformation (default: 3.0)

# Returns
Transformed scores where:
- Perfect similarity (1.0) maps to 0.0
- Lower similarities produce exponentially larger positive values
"""
function exp_score(similarity::AbstractArray; scale::Real = 3.0f0)
    return exp.((1.0f0 .- similarity) .* scale) .- 1.0f0
end

"""
    z_score(phases::AbstractArray) -> AbstractArray

Compute a z-score like metric for phase values using inverse hyperbolic tangent.
Provides a measure of how far phases deviate from the midpoint (0.5).

# Arguments
- `phases`: Array of phase values in [-1, 1] range

# Implementation
1. Centers phases around 0 by subtracting 0.5
2. Remaps to standard phase range
3. Applies inverse hyperbolic tangent and takes absolute value

# Returns
Non-negative scores that grow larger for phases further from 0.5
"""
function z_score(phases::AbstractArray)
    arc = remap_phase(phases .- 0.5f0)
    score = abs.(atanh.(arc))
    return score
end

"""
    similarity_correlation(static_similarity::Matrix{<:Real}, dynamic_similarity::Array{<:Real,3}) -> Vector{<:Real}

Compute correlation between static similarity matrix and each time step of dynamic similarities.

# Arguments
- `static_similarity`: Reference similarity matrix (2D)
- `dynamic_similarity`: Time-varying similarity matrices (3D array with time as third dimension)

# Returns
Vector of correlation values, one for each time step
"""
function similarity_correlation(static_similarity::Matrix{<:Real}, dynamic_similarity::Array{<:Real,3})
    n_steps = axes(dynamic_similarity, 3)
    cor_vals = [cor_realvals(static_similarity |> vec, dynamic_similarity[:,:,i] |> vec) for i in n_steps]
    return cor_vals
end

"""
    cycle_correlation(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3}) -> Vector{<:Real}

Compute correlation between static phase pattern and each cycle of dynamic phases.

# Arguments
- `static_phases`: Reference phase matrix (2D)
- `dynamic_phases`: Time-varying phase matrices (3D array with cycles as third dimension)

# Returns
Vector of correlation values, one for each cycle

See also: [`cor_realvals`](@ref) for handling NaN values in correlation
"""
function cycle_correlation(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3})
    n_cycles = axes(dynamic_phases, 3)
    cor_vals = [cor_realvals(static_phases |> vec, dynamic_phases[:,:,i] |> vec) for i in n_cycles]
    return cor_vals
end

"""
    cycle_sparsity(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3}) -> Vector{<:Real}

Calculate the sparsity (proportion of NaN values) for each cycle of dynamic phases.

# Arguments
- `static_phases`: Reference phase matrix (2D), used for size reference
- `dynamic_phases`: Time-varying phase matrices (3D array with cycles as third dimension)

# Returns
Vector of sparsity values (0.0-1.0) for each cycle, where:
- 0.0 indicates no NaN values
- 1.0 indicates all values are NaN
"""
function cycle_sparsity(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3})
    n_cycles = axes(dynamic_phases, 3)
    total = reduce(*, size(static_phases))
    sparsity_vals = [sum(isnan.(dynamic_phases[:,:,i])) / total for i in n_cycles]
    return sparsity_vals
end

"""
    cor_realvals(x, y) -> Real

Compute correlation between two arrays, handling NaN values appropriately.
Only considers positions where both arrays have real (non-NaN) values.

# Arguments
- `x, y`: Arrays of same size, potentially containing NaN values

# Returns
- Correlation coefficient between non-NaN values
- Returns 0.0 if no valid pairs of values exist

Used by [`cycle_correlation`](@ref) and [`similarity_correlation`](@ref)
"""
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
    
"""
    interpolate_roc(roc) -> Function

Create an interpolated function from ROC curve points.
Useful for smooth ROC curve visualization and AUC calculation.

# Arguments
- `roc`: Tuple of (TPR, FPR) vectors from [`tpr_fpr`](@ref)

# Returns
Interpolation function mapping FPR to TPR values
"""
function interpolate_roc(roc)
    tpr, fpr = roc
    interp = linear_interpolation(fpr, tpr)
    return interp
end

"""
    dense_onehot(x::OneHotMatrix) -> Matrix{Float32}

Convert a OneHotMatrix to a dense Float32 matrix.
Useful for compatibility with neural network operations.

# Arguments
- `x`: OneHotMatrix representation

# Returns
Dense matrix with 1.0f0 values where x was true
"""
function dense_onehot(x::OneHotMatrix)
    return 1.0f0 .* x
end

"""
    confusion_matrix(sim, truth, threshold::Real) -> Matrix{<:Real}

Compute binary confusion matrix given similarity scores and true labels.

# Arguments
- `sim`: Similarity or prediction scores
- `truth`: True labels (binary)
- `threshold`: Decision threshold for converting scores to predictions

# Returns
2×2 confusion matrix with format:
[True Positive  False Positive]
[False Negative True Negative]

Used by [`OvR_matrices`](@ref) for multi-class evaluation
"""
function confusion_matrix(sim, truth, threshold::Real)
    truth = hcat(truth .== 1, truth .== 0)
    prediction = hcat(sim .> threshold, sim .<= threshold)

    confusion = truth' * prediction
    return confusion
end


#Loss functions

"""
    quadrature_loss(phases::AbstractArray, truth::AbstractArray; dim::Int = 1) -> AbstractArray

Compute loss based on quadrature phase encoding relative to target values.

# Arguments
- `phases`: Array of phase values
- `truth`: Target values (binary/one-hot encoded)
- `dim`: Dimension along which to compute similarity (default: 1)

# Implementation
1. Scales truth values to target phases (0.5 * truth)
2. Computes similarity between phases and targets
3. Returns dissimilarity (1 - similarity)

Used for phase-based classification tasks.
"""
function quadrature_loss(phases::AbstractArray, truth::AbstractArray; dim::Int = 1)
    targets = 0.5f0 .* truth
    sim = similarity(phases, targets, dim = dim)
    return 1.0f0 .- sim
end

"""
    similarity_loss(similarities::AbstractArray, truth::AbstractArray; dim::Int = 1) -> AbstractArray

Compute a smooth loss function based on phase similarities and truth values.

# Arguments
- `similarities`: Array of similarity values
- `truth`: Target values (binary/one-hot encoded)
- `dim`: Dimension along which to compute loss (default: 1)

# Implementation
1. Computes absolute difference from perfect similarity (1.0)
2. Weights differences by truth values
3. Applies smooth sin² transformation for stable gradients

Returns a loss that:
- Is 0 for perfect similarity with correct class
- Grows smoothly for increasing dissimilarity
- Is properly scaled for gradient-based optimization
"""
function similarity_loss(similarities::AbstractArray, truth::AbstractArray; dim::Int = 1)
    distance = abs.(1.0 .- similarities) .* truth
    distance = sum(distance .* truth, dims = dim)
    loss = 2.0f0 .* sin.(pi_f32/4.0f0 .* distance) .^ 2.0f0
    return loss
end

"""
    evaluate_loss(predictions::AbstractArray, truth::AbstractArray, encoding::Symbol = :similarity; reduce_dim::Int = 1) -> AbstractArray
    evaluate_loss(predictions::SpikingCall, truth::AbstractArray, encoding::Symbol = :similarity; reduce_dim::Int = 1) -> AbstractArray

Evaluate loss between predictions and truth values using specified encoding scheme.

# Arguments
- `predictions`: Model predictions (phases or spike trains)
- `truth`: Target values
- `encoding`: Encoding scheme (:similarity or :quadrature)
- `reduce_dim`: Dimension along which to compute loss (default: 1)

# Encoding Schemes
- `:similarity`: Uses [`similarity_loss`](@ref)
- `:quadrature`: Uses [`quadrature_loss`](@ref)

# Implementation
1. Selects appropriate loss function based on encoding
2. Handles dimension mismatches between predictions and truth
3. For spike trains, converts to phases before evaluation

Returns loss values appropriate for the chosen encoding scheme.
"""
function evaluate_loss(predictions::AbstractArray, truth::AbstractArray, encoding::Symbol = :similarity; reduce_dim::Int = 1)
    if encoding == :quadrature
        loss_fn = quadrature_loss
    else
        loss_fn = similarity_loss
    end

    #match the loss dispatch dimensions against the truth & predictions
    n_d_pred = ndims(predictions)
    n_d_truth = ndims(truth)
    if n_d_pred == 2 && n_d_truth == 2
        return loss_fn(predictions, truth, dim=reduce_dim)
    else
        dispatch_dims = setdiff(Set(1:n_d_pred), Set(1:n_d_truth)) |> Tuple
        return map(x -> loss_fn(x, truth, dim=reduce_dim), eachslice(predictions, dims=dispatch_dims))
    end
end

function evaluate_loss(predictions::SpikingCall, truth::AbstractArray, encoding::Symbol = :similarity; reduce_dim::Int = 1)
    predictions = train_to_phase(predictions) 
    truth = truth
    return evaluate_loss(predictions, truth, encoding, reduce_dim=reduce_dim)
end

# Prediction functions
"""
    predict_quadrature(phases::AbstractArray; dim::Int=1) -> AbstractArray

Convert phase values to class predictions using quadrature encoding.
Finds classes by minimum distance to target phase (0.5).

# Arguments
- `phases`: Array of phase values
- `dim`: Dimension along which to find predictions (default: 1)

# Implementation
Automatically handles GPU arrays by moving to CPU for argmin operation.
"""
function predict_quadrature(phases::AbstractArray; dim::Int=1)
    if on_gpu(phases)
        phases = phases |> cdev
    end

    predictions = getindex.(argmin(abs.(phases .- 0.5f0), dims=dim), dim)
    return predictions
end

"""
    predict_similarity(sims::AbstractArray; dim::Int=1) -> AbstractArray

Convert similarity values to class predictions by taking argmax.
Used for similarity-based classification.

# Arguments
- `sims`: Array of similarity values
- `dim`: Dimension along which to find maximum (default: 1)

# Implementation
Automatically handles GPU arrays by moving to CPU for argmax operation.
"""
function predict_similarity(sims::AbstractArray; dim::Int=1)
    if on_gpu(sims)
        sims = sims |> cdev
    end

    predictions = vec(getindex.(argmax(sims, dims=dim), dim))
    return predictions
end

"""
    predict(predictions::AbstractArray, encoding::Symbol = :similarity; reduce_dim=1) -> AbstractArray
    predict(predictions::SpikingCall, encoding::Symbol = :similarity; reduce_dim::Int=1) -> AbstractArray

Convert model outputs to class predictions using specified encoding scheme.

# Arguments
- `predictions`: Model outputs (phases, similarities, or spike trains)
- `encoding`: Encoding scheme (:similarity or :quadrature)
- `reduce_dim`: Dimension along which to make predictions

# Encoding Schemes
- `:similarity`: Uses [`predict_similarity`](@ref)
- `:quadrature`: Uses [`predict_quadrature`](@ref)

For spike trains, automatically converts to phases before prediction.
"""
function predict(predictions::AbstractArray, encoding::Symbol = :similarity; reduce_dim=1)
    if encoding == :quadrature
        predict_fn = x -> predict_quadrature(x, dim=reduce_dim)
    else
        predict_fn = x -> predict_similarity(x, dim=reduce_dim)
    end

    return predict_fn(predictions)
end

function predict(predictions::SpikingCall, encoding::Symbol = :similarity; reduce_dim::Int=1)
    predictions = train_to_phase(predictions)
    return predict(predictions, encoding, reduce_dim=reduce_dim)
end

# Performance evaluation functions
"""
    evaluate_accuracy(values::AbstractArray, truth::AbstractArray, encoding::Symbol; reduce_dim::Int=1) -> Tuple{Union{Int,Array},Int}
    evaluate_accuracy(values::SpikingCall, truth::AbstractArray, encoding::Symbol; reduce_dim::Int=1) -> Tuple{Union{Int,Array},Int}

Evaluate classification accuracy for phase-based or spiking neural networks.

# Arguments
- `values`: Model predictions (phases, similarities, or spike trains)
- `truth`: Ground truth labels in one-hot format
- `encoding`: Encoding scheme (:similarity or :quadrature)
- `reduce_dim`: Dimension along which to compute accuracy (default: 1)

# Implementation
1. Moves data to CPU if on GPU
2. Finds indices of true classes from one-hot truth
3. Converts predictions to class indices using specified encoding
4. Counts correct predictions
5. For spike trains, first converts to phases

# Returns
Tuple of (correct_predictions, total_samples) where:
- `correct_predictions`: Number of correct predictions (or array for multiple evaluations)
- `total_samples`: Total number of samples evaluated

# Notes
- Handles both direct predictions and spike train inputs
- Supports different dimensionality between predictions and truth
- Automatically handles GPU/CPU device placement

See also: [`predict`](@ref) for the underlying prediction mechanism
"""
function evaluate_accuracy(values::AbstractArray, truth::AbstractArray, encoding::Symbol; reduce_dim::Int=1)
    if on_gpu(values, truth)
        values = values |> cdev
        truth = truth |> cdev
    end

    @assert ndims(values) >= ndims(truth) "Dimensionality of truth must be able to map onto values"
    reshape_dims = [d == reduce_dim ? 1 : size(truth,d) for d in 1:ndims(truth)]
    idx = reshape(getindex.(findall(truth .== 1.0f0), reduce_dim), reshape_dims...)
    predict_fn = x -> sum(predict(x, encoding, reduce_dim=reduce_dim) .== idx)
    n_truth = prod([size(truth, d) for d in setdiff(Set(1:ndims(truth)), Set(reduce_dim))])

    if ndims(values) > ndims(truth)
        dispatch_dims = setdiff(Set(1:ndims(values)), Set(1:ndims(truth))) |> Tuple
        response = map(predict_fn, eachslice(values, dims=dispatch_dims))
    else
        response = predict_fn(values)
    end
    
    return response, n_truth
end

function evaluate_accuracy(values::SpikingCall, truth::AbstractArray, encoding::Symbol; reduce_dim::Int=1)
    values = train_to_phase(values)
    return evaluate_accuracy(values, truth, encoding, reduce_dim=reduce_dim)
end

"""
    loss_and_accuracy(data_loader, model, ps, st, args; reduce_dim::Int=1, encoding::Symbol = :codebook) -> Tuple{Real,Real}

Evaluate model loss and accuracy on a dataset.

# Arguments
- `data_loader`: Iterator providing (input, target) pairs
- `model`: Neural network model
- `ps`: Model parameters
- `st`: Model state
- `args`: Configuration arguments (including use_cuda)
- `reduce_dim`: Dimension for reduction operations
- `encoding`: Phase encoding scheme (:codebook or :quadrature)

# Returns
Tuple of (average_loss, accuracy)

# Notes
Automatically handles GPU/CPU device placement based on args.use_cuda
"""
function loss_and_accuracy(data_loader, model, ps, st, args; reduce_dim::Int=1, encoding::Symbol = :codebook)
    loss_fn = (x, y) -> evaluate_loss(x, y, encoding, reduce_dim=reduce_dim)

    if args.use_cuda && CUDA.functional()
        dev = gdev
    else
        dev = cdev
    end

    num = 0
    correct = 0
    ls = 0.0f0

    for (x, y) in data_loader
        x = x |> dev
        y = y |> dev
        ŷ, _ = model(x, ps, st)
        @assert typeof(ŷ) != SpikingCall "Must call spiking models with SpikingArgs provided"

        ls += sum(stack(cdev(loss_fn(ŷ, y)))) #sum across batches
        model_correct, answers = cdev.(evaluate_accuracy(ŷ, y, encoding, reduce_dim=reduce_dim))
        correct += model_correct
        num += answers
    end

    return ls / num, correct / num
end

"""
    spiking_loss_and_accuracy(data_loader, model, ps, st, args; reduce_dim::Int=1, encoding::Symbol = :codebook, repeats::Int) -> Tuple{Array,Array}

Evaluate spiking neural network loss and accuracy over multiple repeats.

# Arguments
- `data_loader`: Iterator providing (input, target) pairs
- `model`: Spiking neural network model
- `ps`: Model parameters
- `st`: Model state
- `args`: Configuration arguments
- `reduce_dim`: Dimension for reduction operations
- `encoding`: Phase encoding scheme
- `repeats`: Number of evaluation repeats

# Returns
Tuple of (losses, accuracies) where:
- losses: Array of shape (1, repeats)
- accuracies: Array of length repeats

Useful for assessing the reliability of spiking network performance.
"""
function spiking_loss_and_accuracy(data_loader, model, ps, st, args; reduce_dim::Int=1, encoding::Symbol = :codebook, repeats::Int)
    loss_fn = (x, y) -> evaluate_loss(x, y, encoding, reduce_dim=reduce_dim)

    if args.use_cuda && CUDA.functional()
        dev = gdev
    else
        dev = cdev
    end

    num = 0
    correct = zeros(Int64, repeats)
    ls = zeros(Float32, (1,repeats))

    for (x, y) in data_loader
        x = x |> dev
        y = y |> dev
        ŷ, _ = model(x, ps, st)
        ls .+= sum(stack(cdev(loss_fn(ŷ, y))), dims=1) #sum across batches
        model_correct, answers = cdev.(evaluate_accuracy(ŷ, y, encoding, reduce_dim=reduce_dim))
        correct .+= model_correct
        num += answers
    end

    return ls ./ num, correct ./ num
end