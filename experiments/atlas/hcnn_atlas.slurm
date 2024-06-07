#!/bin/bash
#SBATCH --job-name=atlas
#SBATCH --account=spe
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --export=all
#SBATCH -e hcnn_atlas.err
#SBATCH -o hcnn_atlas.log

model_version_list='HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50'
# model_version_list='HCNN_biopython_proteinnet_extra_mols_0p00'
# model_version_list='HCNN_biopython_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p50 HCNN_pyrosetta_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50 HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
use_mt_structure_list='0 1'

pdb_dir='/gscratch/spe/gvisan01/tcr_pmhc/pdbs/ATLAS/'

base_dir='./'

for model_version in $model_version_list
    do
    for use_mt_structure in $use_mt_structure_list
        do

        python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                            --model_version $model_version \
                            --csv_file $base_dir'ATLAS_cleaned.csv' \
                            --folder_with_pdbs $pdb_dir \
                            --output_dir $base_dir \
                            --wt_pdb_column wt_pdb \
                            --mt_pdb_column mt_pdb \
                            --mutant_column mutant \
                            --mutant_chain_column chain \
                            --mutant_split_symbol"=|" \
                            --use_mt_structure $use_mt_structure
        
        python -u correlations.py \
                    --model_version $model_version \
                    --use_mt_structure $use_mt_structure
        
    done
done
