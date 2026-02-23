# PhasorNetworks.jl - GPU-capable Docker image
#
# CUDA.jl 5.x bundles its own CUDA toolkit via JLL artifacts, so no CUDA
# base image is required. GPU access is provided at runtime via:
#   docker run --gpus all <image>
# (requires NVIDIA Container Toolkit on the host)

FROM julia:1.11

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY Project.toml Manifest.toml ./

# Instantiate dependencies (downloads all packages + CUDA artifacts)
# This layer is cached as long as the manifests don't change
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Copy source and test files
COPY src/ ./src/
COPY test/ ./test/

# Precompile the package and its dependencies
# CUDA.jl can precompile without a GPU present; GPU-specific code is JIT-compiled at runtime
RUN julia --project=. -e 'using Pkg; Pkg.precompile()'

# Run the full test suite by default
# Tests auto-detect CUDA via CUDA.functional() and run GPU tests if a device is available
CMD ["julia", "--project=.", "-e", "using Pkg; Pkg.test()"]
