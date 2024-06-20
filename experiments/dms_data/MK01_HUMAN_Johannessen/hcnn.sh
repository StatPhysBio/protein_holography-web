

# model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50 HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 '
model_version_list='HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50'
# model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50 HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
pdb_dir='./pdbs/'

base_dir='./'

systems='output_MK01_HUMAN_Johannessen__4g6n output_MK01_HUMAN_Johannessen__AF-P28482-F1-model_v4'

dms_columns='ETP_AVERAGE DOX_Average SCH_Average VRT_AVERAGE'

for system in $systems
    do
    for model_version in $model_version_list
        do

        python -u ../../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                            --model_version $model_version \
                            --csv_file $base_dir$system'.csv' \
                            --folder_with_pdbs $pdb_dir \
                            --output_dir $base_dir \
                            --wt_pdb_column wt_pdb \
                            --mutant_column mutant \
                            --mutant_chain_column chain \
                            --use_mt_structure 0 \
                            --dms_column $dms_columns


    python -u ../correlations.py \
                    --model_version $model_version \
                    --use_mt_structure 0 \
                    --system_name $system \
                    --dms_column $dms_columns
    
    done
done
