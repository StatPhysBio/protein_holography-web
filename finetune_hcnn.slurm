#!/bin/bash
#SBATCH --job-name=finetune_hcnn
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e finetune_hcnn.err
#SBATCH -o finetune_hcnn.log

python finetune_hcnn.py \
            -c ./training_data/finetuning/rosetta_ddg/configs/HCNN_pyrosetta_proteinnet_extra_mols_0p50_with_0p50_all.yaml \
            -i 9

