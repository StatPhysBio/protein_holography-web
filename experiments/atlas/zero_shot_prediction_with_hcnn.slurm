#!/bin/bash
#SBATCH --job-name=atlas
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e zero_shot_prediction_with_hcnn.err
#SBATCH -o zero_shot_prediction_with_hcnn.log

bash zero_shot_prediction_with_hcnn.sh
