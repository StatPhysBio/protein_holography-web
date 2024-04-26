

model_version_list='HCNN_0p00' #'HCNN_0p00 HCNN_0p50'
use_mt_structure_list='0 1'

pdb_dir='/gscratch/spe/gvisan01/tcr_pmhc/pdbs/ATLAS/'

base_dir='./'

for model_version in $model_version_list
    do
    for use_mt_structure in $use_mt_structure_list
        do

        python -u ../../zero_shot_mutation_effect_prediction_with_hcnn.py \
                            --model_version $model_version \
                            --csv_file $base_dir'ATLAS_cleaned.csv' \
                            --folder_with_pdbs $pdb_dir \
                            --output_dir $base_dir \
                            --dms_column '[ log10(Kd_wt/Kd_mt) ]' \
                            --mutant_column mutant \
                            --mutant_chain_column chain \
                            --mutant_split_symbol"=|" \
                            --use_mt_structure $use_mt_structure
        
    done
done
