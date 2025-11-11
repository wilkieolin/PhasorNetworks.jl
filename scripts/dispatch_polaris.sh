#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A EE-ECP

PROJ_DIR="/eagle/EE-ECP/"
CODE_PATH="${PROJ_DIR}PhasorNetworks.jl"

EPOCHS=5
OPTIMIZER="rmsprop"
LEARNING_RATE=0.001
BATCHSIZE=128
USE_CUDA="true"

module use /eagle/EE-ECP/julia_depot/modulefiles/
module load julia/1.11

cd $CODE_PATH

julia scripts/train_fashionmnist.jl --lr $LEARNING_RATE \
     --epochs $EPOCHS \
     --optimizer $OPTIMIZER \
     --batchsize $BATCHSIZE \
     --use_cuda $USE_CUDA