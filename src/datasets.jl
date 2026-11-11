using Downloads: download
using CodecZlib: GzipDecompressorStream
using Scratch: @get_scratch!

const _FMNIST_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
const _FMNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

function _fmnist_path(name::AbstractString)
    dir = @get_scratch!("fashion_mnist")
    path = joinpath(dir, name)
    if !isfile(path)
        url = "$(_FMNIST_BASE)/$(name)"
        @info "Downloading $name" url
        download(url, path)
    end
    return path
end

function _read_idx_gz(path::AbstractString)
    open(path) do f
        gz = GzipDecompressorStream(f)
        try
            magic = ntoh(read(gz, UInt32))
            n = Int(ntoh(read(gz, UInt32)))
            if magic == 0x00000803
                rows = Int(ntoh(read(gz, UInt32)))
                cols = Int(ntoh(read(gz, UInt32)))
                buf = read(gz)
                length(buf) == rows * cols * n ||
                    error("FashionMNIST image file truncated: got $(length(buf)) bytes, expected $(rows*cols*n)")
                # IDX is row-major, image-major: pixel(i,r,c) at offset (i-1)*rows*cols + (r-1)*cols + (c-1)
                # reshape col-major as (cols, rows, n), then permute → (rows, cols, n)
                return permutedims(reshape(buf, (cols, rows, n)), (2, 1, 3))
            elseif magic == 0x00000801
                buf = read(gz)
                length(buf) == n ||
                    error("FashionMNIST label file truncated: got $(length(buf)) bytes, expected $n")
                return buf
            else
                error("Unknown IDX magic number: 0x$(string(magic, base=16, pad=8))")
            end
        finally
            close(gz)
        end
    end
end

"""
    fashion_mnist_data(split::Symbol) -> (features, targets)

Load Fashion-MNIST. Downloads the gzipped IDX files on first call and caches them
in a Scratch.jl scratchspace owned by PhasorNetworks.

# Arguments
- `split` — `:train` (60000 examples) or `:test` (10000 examples).

# Returns
A NamedTuple `(features, targets)`:
- `features::Array{Float32, 3}` — shape `(28, 28, N)`, normalized to `[0, 1]`.
- `targets::Vector{Int}` — class labels in `0:9`.

The shape and normalization match `MLDatasets.FashionMNIST(split=split)` so
existing callsites that use `.features` / `.targets` keep working.
"""
function fashion_mnist_data(split::Symbol)
    prefix = split === :train ? "train" :
             split === :test  ? "t10k"  :
             error("split must be :train or :test, got $(repr(split))")
    images = _read_idx_gz(_fmnist_path("$(prefix)-images-idx3-ubyte.gz"))
    labels = _read_idx_gz(_fmnist_path("$(prefix)-labels-idx1-ubyte.gz"))
    features = Float32.(images) ./ 255f0
    targets = Int.(labels)
    return (; features, targets)
end
