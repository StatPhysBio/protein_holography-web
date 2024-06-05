

model_version_list='HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50 HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
use_mt_structure_list='0 1'

base_dir='./'

for model_version in $model_version_list
    do
    for use_mt_structure in $use_mt_structure_list
        do

        python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                            --model_version $model_version \
                            --csv_file $base_dir'T4_mutant_ddG_standardized.csv' \
                            --folder_with_pdbs $base_dir'pdbs' \
                            --output_dir $base_dir \
                            --dms_column ddG \
                            --dms_label None \
                            --wt_pdb_column wt_pdb \
                            --mt_pdb_column mt_pdb \
                            --mutant_column mutant \
                            --mutant_chain_column mutant_chain \
                            --use_mt_structure $use_mt_structure
        
        python -u correlations.py \
                            --model_version $model_version \
                            --use_mt_structure $use_mt_structure
    
    done
done