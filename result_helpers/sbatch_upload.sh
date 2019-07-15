#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=1-00:00
#SBATCH --job-name myfirst
module load arch/avx512 StdEnv/2018.3
nvidia-smi