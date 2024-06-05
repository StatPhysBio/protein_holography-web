


model_version_list='HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00' # HCNN_pyrosetta_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50 HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
use_mt_structure='0'

base_dir='./'
output_dir=$base_dir

for model_version in $model_version_list
    do

    echo $model_version

    python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                        --model_version $model_version \
                        --csv_file $base_dir's669_ddg_experimental.csv' \
                        --folder_with_pdbs $base_dir'pdbs/' \
                        --output_dir $output_dir \
                        --wt_pdb_column pdbid \
                        --mutant_column variant \
                        --mutant_chain_column chainid \
                        --mutant_split_symbol"=|" \
                        --use_mt_structure $use_mt_structure
    
    python -u correlations.py \
                        --model_version $model_version \
                        --use_mt_structure $use_mt_structure \
                        --system_name s669

done

