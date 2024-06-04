

# model_version_list='HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50 HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50' # HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
model_version_list='HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
use_mt_structure='0'


for model_version in $model_version_list
    do

    echo $model_version

    python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                        --model_version $model_version \
                        --csv_file Ssym_dir/ssym_dir_ddg_experimental.csv \
                        --folder_with_pdbs Ssym_dir/pdbs/ \
                        --output_dir Ssym_dir \
                        --wt_pdb_column pdbid \
                        --mutant_column variant \
                        --mutant_chain_column chainid \
                        --mutant_split_symbol"=|" \
                        --use_mt_structure $use_mt_structure
    
    python -u correlations.py \
                --dataset Ssym_dir \
                --model_version $model_version \
                --use_mt_structure $use_mt_structure

    python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                        --model_version $model_version \
                        --csv_file Ssym_inv/ssym_inv_ddg_experimental.csv \
                        --folder_with_pdbs Ssym_inv/pdbs/ \
                        --output_dir Ssym_inv \
                        --wt_pdb_column pdbid \
                        --mutant_column variant \
                        --mutant_chain_column chainid \
                        --mutant_split_symbol"=|" \
                        --use_mt_structure $use_mt_structure
    
    python -u correlations.py \
                --dataset Ssym_inv \
                --model_version $model_version \
                --use_mt_structure $use_mt_structure
    
    python -u compute_antisymmetry_score.py \
                --model_version $model_version \
                --use_mt_structure $use_mt_structure

done
