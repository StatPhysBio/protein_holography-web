

# model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50 HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 '
# model_version_list='HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_invariant_mlp HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all'
# model_version_list='HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all'
model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all'
# model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50 HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
pdb_dir='./pdbs/'

base_dir='./'

systems='output_KKA2_KLEPN_Mikkelsen2014__1nd4 output_KKA2_KLEPN_Mikkelsen2014__AF-P00552-F1-model_v4'

dms_columns='Ami11_avg Ami12_avg Ami14_avg Ami18_avg G41811_avg G41812_avg G41814_avg Kan11_avg Kan12_avg Kan14_avg Kan18_avg Neo11_avg Neo12_avg Neo14_avg Neo18_avg Paro11_avg Paro12_avg Paro14_avg Paro18_avg Ribo11_avg Ribo12_avg Ribo14_avg Ribo18_avg'

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
