

# model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50 HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 '
model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50 HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
pdb_dir='./pdbs/'

base_dir='./'

systems='luksza_et_al_tcr1_ec50_sat_mut luksza_et_al_tcr2_ec50_sat_mut luksza_et_al_tcr3_ec50_sat_mut luksza_et_al_tcr4_ec50_sat_mut luksza_et_al_tcr5_ec50_sat_mut luksza_et_al_tcr6_ec50_sat_mut luksza_et_al_tcr7_ec50_sat_mut'

for system in $systems
    do
    for model_version in $model_version_list
        do

        python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                            --model_version $model_version \
                            --csv_file $base_dir$system'.csv' \
                            --folder_with_pdbs $pdb_dir \
                            --output_dir $base_dir \
                            --wt_pdb_column wt_pdb \
                            --mutant_column mutant \
                            --mutant_chain_column mutant_chain \
                            --use_mt_structure 0
        
        python -u plots.py \
                    --model_version $model_version \
                    --use_mt_structure 0
    
    done
done
