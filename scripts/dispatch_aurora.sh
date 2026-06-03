#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A EE-ECP

# Aurora training dispatch — modeled on dispatch_polaris.sh but with
# Aurora's filesystem, module paths, and project location (from
# aurora_test.sh). Targets the oneAPI training script, which loads
# `using oneAPI` to activate PhasorNetworksOneAPIExt.

PROJ_DIR="/flare/EE-ECP/wolin/"
CODE_PATH="${PROJ_DIR}PhasorNetworks.jl"

EPOCHS=5
OPTIMIZER="rmsprop"
LEARNING_RATE=0.001
BATCHSIZE=128
USE_GPU="true"

module use /soft/modulefiles
module load libraries/julia/1.12

cd $CODE_PATH

julia --project=. scripts/train_fashionmnist_aurora.jl --lr $LEARNING_RATE \
     --epochs $EPOCHS \
     --optimizer $OPTIMIZER \
     --batchsize $BATCHSIZE \
     --use_gpu $USE_GPU
