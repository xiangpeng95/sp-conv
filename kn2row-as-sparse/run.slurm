#!/bin/bash

#SBATCH -o job_%j_%N.out
#SBATCH --partition=GPU
#SBATCH -J xpxpxpxppxpxpxpxpxppxpxpxpxpp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 1010:00

./mat_mul
 

