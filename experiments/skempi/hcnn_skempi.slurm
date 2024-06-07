#!/bin/bash
#SBATCH --job-name=skempi
#SBATCH --account=spe
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=32G
#SBATCH --export=all
#SBATCH -e hcnn_skempi.err
#SBATCH -o hcnn_skempi.log

model_version_list='HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50'
# model_version_list='HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50'
# model_version_list='HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
use_mt_structure='0'

base_dir='./'

for model_version in $model_version_list
    do

    python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                        --model_version $model_version \
                        --csv_file $base_dir'skempi_v2_cleaned_NO_1KBH.csv' \
                        --folder_with_pdbs /gscratch/stf/gvisan01/skempi/pyrosetta_mutated_pdbs \
                        --output_dir $base_dir \
                        --wt_pdb_column PDB_filename \
                        --mt_pdb_column PDB_filename_pyrosetta_mutant \
                        --mutant_column mutant \
                        --mutant_chain_column mutant_chain \
                        --mutant_split_symbol"=|" \
                        --use_mt_structure $use_mt_structure
    
    python -u correlations.py \
                        --model_version $model_version \
                        --use_mt_structure $use_mt_structure

done
