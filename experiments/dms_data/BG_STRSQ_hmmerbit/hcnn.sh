

# model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p50 HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p50 '
# model_version_list='HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_invariant_mlp HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all'
# model_version_list='HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all'
model_version_list='HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all'
# model_version_list='HCNN_biopython_pisces30_0p00 HCNN_biopython_pisces30_0p50 HCNN_biopython_pisces90_0p00 HCNN_biopython_pisces90_0p50'
pdb_dir='./pdbs/'

base_dir='./'

systems='output_BG_STRSQ_hmmerbit__1gnx output_BG_STRSQ_hmmerbit__1gon output_BG_STRSQ_hmmerbit__rank_1_model_4_ptm_seed_0_unrelaxed'

dms_columns='enrichment'

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
