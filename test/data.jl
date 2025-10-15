"""
Real-valued, synthetic data
"""

function bullseye_data(n_s::Int, rng::AbstractRNG)
    d = Normal(0.0, 0.08)
    #determine the class labels
    y = rand(rng, (0, 1), n_s)
    #determine the polar coordinates
    r = rand(rng, d, n_s) .+ (0.4 .* y)
    phi = (rand(rng, Float64, n_s) .- 1) .* (2 * pi)
    #convert to cartesian
    x_x = r .* cos.(phi)
    x_y = r .* sin.(phi)

    data = Float32.(cat(x_x, x_y, dims=2)' )
    labels = onehotbatch(y, 0:1)

    return data, labels
end

function generate_helix(r::Real, theta::Real, frequency::Real, scale::Real)
    """
    Construct a generating function for a helix
    
    Parameters:
    - r:       Cylinder radius (distance from z-axis)
    - theta:   Initial angular phase offset [radians]
    - frequency:       Angular frequency [radians/unit time]
    - scale:   How far it stretches in z per unit time [1/unit time]
    
    Returns:
    - Array (x, y, z) of coordinate vectors
    """
    function helix(t::Real)
        v = r * exp(1im * (theta + frequency * t))
        x = real(v)
        y = imag(v)
        z = scale * t
        return [x,y,z]
    end
    return helix
end

function helix_data(n_points::Int)
    #setup the generating functions
    left = generate_helix(1.0, 0.0, 10.0, 1.0)
    right = generate_helix(1.0, pi, 10.0, 1.0)
    #choose the parametric sampling points
    ts = range(start = 0.0, stop = 1.0, length = n_points)
    #generate the coordinates
    left_pts = left.(ts) |> stack
    right_pts = right.(ts) |> stack
    pts = cat(left_pts, right_pts, dims=2)
    #label what belongs to which function
    labels = cat(zeros(n_points), ones(n_points), dims=1)
    
    return pts, labels
end

"""
HD / Attention data
"""
function generate_codebook(rng::AbstractRNG; vocab_size::Int=100, n_hd::Int=512)
    symbols = random_symbols(rng, (n_hd, vocab_size))
    codebook = Dict{Int, Vector{<:Real}}()
    for i in 1:vocab_size
        codebook[i] = symbols[:,i]
    end
    codebook[Int(0)] = zeros(n_hd)
    return codebook
end

function map_symbols(dataset::Vector{<:Any}, codebook::Dict{<:Int, <:Vector{<:Real}})
    map_fn = x -> stack([codebook[k] for k in x])

    output = [map_fn.(data) for data in dataset]
    return output
end

function generate_addresses(n_samples::Int,  n_vsa::Int, rng::AbstractRNG)
    header = random_symbols(rng, (n_vsa, 1))
    powers = collect(0:n_samples-1)
    addresses = [v_bind(header, header .* p)[:,1] for p in powers]
    addresses = stack(addresses, dims=2)
    return header, addresses
end

function generate_copy_dataset(rng::AbstractRNG; num_samples::Int=1000, max_length::Int=50, vocab_size::Int=100)
    dataset = []
    for _ in 1:num_samples
        length = rand(rng, 5:max_length)
        sequence = [rand(1:vocab_size) for _ in 1:length]
        #pad with zeros
        sequence = cat(sequence, zeros(Int, max_length - length), dims=1)
        push!(dataset, (sequence, sequence))  # Input and target identical
    end
    return dataset
end

function generate_reversal_dataset(rng::AbstractRNG; num_samples=1000, max_length=50, vocab_size=100)
    dataset = []
    for _ in 1:num_samples
        length = rand(rng, 5:max_length)
        sequence = [rand(rng, 1:vocab_size) for _ in 1:length]
        #pad with zeros
        sequence = cat(sequence, zeros(Int, max_length - length), dims=1)
        reversed_sequence = reverse(sequence)
        push!(dataset, (sequence, reversed_sequence))
    end
    return dataset
end

function generate_retrieval_dataset(rng::AbstractRNG; num_samples=1000, context_length=100, vocab_size=100, special_token=999)
    dataset = []
    for _ in 1:num_samples
        haystack = [rand(rng, 1:vocab_size) for _ in 1:context_length-1]
        needle_position = rand(rng, 1:context_length-1)
        needle_value = rand(rng, 1:vocab_size)
        insert!(haystack, needle_position, needle_value)
        query = vcat(special_token, needle_position)
        target = [haystack[needle_position]]
        push!(dataset, (vcat(haystack, query), target))
    end
    return dataset
end

function generate_sorting_dataset(rng::AbstractRNG; num_samples=1000, max_length=20, vocab_size=100)
    dataset = []
    for _ in 1:num_samples
        length = rand(rng, 5:max_length)
        sequence = [rand(rng, 1:vocab_size) for _ in 1:length]
        sorted_sequence = sort(sequence)
        push!(dataset, (sequence, sorted_sequence))
    end
    return dataset
end

function generate_pattern_dataset(rng::AbstractRNG; num_samples::Int=1000, pattern_length::Int=3, vocab_size::Int=100, special_token::Int=999)
    dataset = []
    for _ in 1:num_samples
        context_length = rand(20:50)
        context = [rand(rng, 1:vocab_size) for _ in 1:context_length]
        pattern = [rand(rng, 1:vocab_size) for _ in 1:pattern_length]
        # Insert pattern at random position
        insert_pos = rand(rng, 1:context_length - pattern_length + 1)
        context[insert_pos:insert_pos+pattern_length-1] = pattern
        push!(dataset, (vcat(context, special_token, pattern[1:end-1]), [pattern[end]]))
    end
    return dataset
end
