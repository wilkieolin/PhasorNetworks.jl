# PhasorNetworks.jl - GPU-capable Docker image
#
# CUDA.jl bundles its own CUDA toolkit via JLL artifacts, so no CUDA
# base image is required. CUDA is a weak dependency of the package; it is
# resolved only in envs/cuda. GPU access is provided at runtime via:
#   docker run --gpus all <image>
# (requires NVIDIA Container Toolkit on the host)

FROM julia:1.12

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY Project.toml Manifest.toml ./
COPY envs/cuda/Project.toml envs/cuda/Manifest.toml ./envs/cuda/

# Copy source, extensions, tests and the backend helper
COPY src/ ./src/
COPY ext/ ./ext/
COPY test/ ./test/
COPY scripts/gpu_backend.jl ./scripts/

# Instantiate + precompile the CUDA environment (downloads packages and
# CUDA artifacts). CUDA.jl precompiles without a GPU present; GPU kernels
# are JIT-compiled at runtime.
RUN julia --project=envs/cuda -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Run the full test suite on the CUDA backend by default
CMD ["julia", "test/gpu.jl", "cuda"]
