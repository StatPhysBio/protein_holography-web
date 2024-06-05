

# model_version_list='HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50 HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00_no_constant'
pdb_dir='./pdbs/'

base_dir='./'


for model_version in $model_version_list
    do

    # python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
    #                     --model_version $model_version \
    #                     --csv_file $base_dir'hsiue_et_al_H2_sat_mut.csv' \
    #                     --folder_with_pdbs $pdb_dir \
    #                     --output_dir $base_dir \
    #                     --wt_pdb_column wt_pdb \
    #                     --mutant_column mutant \
    #                     --mutant_chain_column mutant_chain \
    #                     --use_mt_structure 0

    python -u plots.py \
                --model_version $model_version \
                --use_mt_structure 0

done
