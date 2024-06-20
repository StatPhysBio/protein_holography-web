#!/bin/bash
#SBATCH --job-name=TIM_SULSO_b0
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=44G
#SBATCH -e ./__slurm_logs/TIM_SULSO_b0_proteinmpnn.err
#SBATCH -o ./__slurm_logs/TIM_SULSO_b0_proteinmpnn.out

cd TIM_SULSO_b0
bash proteinmpnn.sh
