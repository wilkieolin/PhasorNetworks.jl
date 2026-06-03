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

julia --project . -e "using Pkg; Pkg.instantiate(); Pkg.test();"