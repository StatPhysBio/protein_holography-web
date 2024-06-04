#!/bin/bash
#SBATCH --job-name=download_pdbs
#SBATCH --account=spe
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --export=all
#SBATCH -e download_pdbs.err
#SBATCH -o download_pdbs.log

python download_pdbs.py --output_dir /gscratch/stf/gvisan01/pisces_pdbs/
