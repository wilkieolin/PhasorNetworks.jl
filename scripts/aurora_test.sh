#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A EE-ECP

PROJ_DIR="/flare/EE-ECP/wolin/"
CODE_PATH="${PROJ_DIR}PhasorNetworks.jl"

module use /soft/modulefiles
module load libraries/julia/1.12

# Aurora's site oneAPI module (loaded by the login profile) puts its own
# libur_adapter_level_zero / libumf on LD_LIBRARY_PATH. LD_LIBRARY_PATH wins
# over the RUNPATH of the libsycl bundled with oneAPI.jl's support library,
# so the site adapters get substituted underneath it and any version
# mismatch aborts the process in sycl::platform::get_platforms
# ("level_zero backend failed with error: 45 (UR_RESULT_ERROR_INVALID_ARGUMENT)",
# SIGABRT) on the first oneMKL call, i.e. the first matmul on a oneArray.
# Drop the site entries so SYCL resolves purely against the bundled stack.
# This is the same workaround oneAPI.jl's own Aurora CI applies
# (JuliaGPU/oneAPI.jl .gitlab-ci.yml, test:lts job).
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/opt/aurora/[^/]*/oneapi/' | paste -sd:)"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

cd $CODE_PATH

# Full suite on the Intel GPUs. test/gpu.jl activates envs/oneapi (the only
# environment that resolves oneAPI), instantiates it, and runs the suite with
# PHASOR_BACKEND=oneapi. Instantiate once on a login node first so the
# compute node (no network) finds everything cached. The GPU section runs
# first, so a device-level failure shows up within minutes rather than after
# the CPU suite.
julia test/gpu.jl oneapi
