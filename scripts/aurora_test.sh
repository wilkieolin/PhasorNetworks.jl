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

cd $CODE_PATH

# Full suite on the Intel GPUs. test/gpu.jl activates envs/oneapi (the only
# environment that resolves oneAPI), instantiates it, and runs the suite with
# PHASOR_BACKEND=oneapi. Instantiate once on a login node first so the
# compute node (no network) finds everything cached.
julia test/gpu.jl oneapi