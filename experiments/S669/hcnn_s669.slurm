#!/bin/bash
#SBATCH --job-name=hcnn_ssym_dir
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e hcnn_ssym_dir.err
#SBATCH -o hcnn_ssym_dir.log

bash hcnn_ssym_dir.sh
