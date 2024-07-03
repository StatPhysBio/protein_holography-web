

proteinmpnn_dir=/gscratch/spe/gvisan01/ProteinMPNN-copy/mutation_effect_prediction/
pdb_dir='./pdbs/'
base_dir='./'

model_version_list='v_48_030 v_48_020 v_48_002'

systems='output_RL401_YEAST_Fraser2016__2g3q output_RL401_YEAST_Fraser2016__AF-P0CH09-F1-model_v4 output_RL401_YEAST_Fraser2016__rank_1_model_4_ptm_seed_0_unrelaxed'

dms_columns='Caffeine DMSO DTT HU MG132'

for system in $systems
    do
    for model_version in $model_version_list
        do

        python -u $proteinmpnn_dir'zero_shot_mutation_effect_prediction__faster.py' \
                        --csv_file $base_dir$system'.csv' \
                        --folder_with_pdbs $pdb_dir \
                        --output_dir $base_dir'proteinmpnn_'$model_version \
                        --use_mt_structure 0 \
                        --model_name $model_version \
                        --num_seq_per_target 10 \
                        --batch_size 10 \
                        --wt_pdb_column wt_pdb \
                        --mutant_column mutant \
                        --mutant_chain_column chain \
        
        python -u ../correlations.py \
                        --model_version proteinmpnn_$model_version \
                        --use_mt_structure 0 \
                        --system_name $system \
                        --dms_column $dms_columns
    
    done
done

