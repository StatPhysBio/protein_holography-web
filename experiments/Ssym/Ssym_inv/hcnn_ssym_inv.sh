


model_version_list='HCNN_proteinnet_0p00 HCNN_proteinnet_0p50' # 'HCNN_0p00 HCNN_0p50'
use_mt_structure='0'

base_dir='./'
output_dir=$base_dir

for model_version in $model_version_list
    do

    python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                        --model_version $model_version \
                        --csv_file $base_dir'ssym_inv_ddg_experimental.csv' \
                        --folder_with_pdbs $base_dir'pdbs/' \
                        --output_dir $output_dir \
                        --wt_pdb_column pdbid \
                        --mutant_column variant \
                        --mutant_chain_column chainid \
                        --mutant_split_symbol"=|" \
                        --use_mt_structure $use_mt_structure

done

