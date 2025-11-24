#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=italoslurm@gmail.com
#SBATCH --job-name=qmc
#SBATCH --output=qmc.out
#SBATCH --error=qmc.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35

cmake --build build

./build/bin/qmc