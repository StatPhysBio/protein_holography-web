

model_version_list='HCNN_biopython_proteinnet_0p00 HCNN_biopython_proteinnet_0p50 HCNN_biopython_proteinnet_extra_mols_0p00 HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all HCNN_biopython_proteinnet_extra_mols_0p50 HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all HCNN_pyrosetta_proteinnet_extra_mols_0p00 HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all HCNN_pyrosetta_proteinnet_extra_mols_0p50 HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all'

for model_version in $model_version_list
    do

    echo $model_version

    python ../../run_hcnn_on_pdbfiles.py \
                        -pd ./pdbs/ \
                        -m $model_version \
                        -r probas logits \
                        -o './output/'$model_version'.csv' \
                        -lb 1 \
                        -v 1

done
