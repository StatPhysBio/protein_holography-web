#!/bin/bash
#SBATCH --job-name=luksza_et_al
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e hcnn_luksza_et_al.err
#SBATCH -o hcnn_luksza_et_al.log

bash hcnn_luksza_et_al.sh
