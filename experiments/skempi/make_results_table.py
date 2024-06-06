
import os, sys
import json
import numpy as np
import pandas as pd


sys.path.append('..')
from get_full_table import METADATA_COLUMNS, MODELS


system_name = 'skempi_v2_cleaned_NO_1KBH'

df = None

for curr_comp_in_json, curr_comp in zip(['Per-Structure', 'Overall'],
                                        ['per structure', 'overall']):
    
    for mut_types, is_single, is_multi in zip(['all_types_of_mutations', 'single_point_mutations', 'multi_point_mutations'],
                                              [True, True, False],
                                              [True, False, True]):    

        correlations_values_in_table = {}

        num_measurements_trace = []
        num_structures_trace = []

        for hcnn_model in MODELS:

            # load json file with correlations
            with open(f'{hcnn_model}/zero_shot_predictions/{system_name}-{hcnn_model}-use_mt_structure=0-correlations.json', 'r') as f:
                correlations = json.load(f)
            
            pr = -correlations[curr_comp_in_json][mut_types]['Pr'] # flip correlation so higher is better
            pr_pval = correlations[curr_comp_in_json][mut_types]['Pr_pval']

            sr = -correlations[curr_comp_in_json][mut_types]['Sr'] # flip correlation so higher is better
            sr_pval = correlations[curr_comp_in_json][mut_types]['Sr_pval']

            num_measurements = correlations[curr_comp_in_json][mut_types]['num']
            num_measurements_trace.append(num_measurements)

            num_structures = correlations[curr_comp_in_json][mut_types]['num_struc']
            num_structures_trace.append(num_structures)

            correlations_values_in_table[hcnn_model + ' - Pearsonr'] = pr
            correlations_values_in_table[hcnn_model + ' - Pearsonr p-value'] = pr_pval
            correlations_values_in_table[hcnn_model + ' - Spearmanr'] = sr
            correlations_values_in_table[hcnn_model + ' - Spearmanr p-value'] = sr_pval

        if len(set(num_measurements_trace)) > 1:
            print('WARNING: Number of measurements for each model is not the same')
            print(num_measurements_trace)
        
        if len(set(num_structures_trace)) > 1:
            print('WARNING: Number of structures for each model is not the same')
            print(num_structures_trace)

        metadata_values = ['SKEMPI 2.0', False, '-ddG binding', num_measurements_trace[-1], num_structures_trace[-1], is_single, is_multi, curr_comp]

        metatadata_in_table = dict(zip(METADATA_COLUMNS, metadata_values))

        if df is None:
            df = pd.DataFrame(metatadata_in_table, index=[0])
            df = pd.concat([df, pd.DataFrame(correlations_values_in_table, index=[0])], axis=1)
        else:
            curr_df = pd.DataFrame(metatadata_in_table, index=[0])
            curr_df = pd.concat([curr_df, pd.DataFrame(correlations_values_in_table, index=[0])], axis=1)
            df = pd.concat([df, curr_df], axis=0)

print(df)
df.to_csv(f'{system_name}-results_table.csv', index=False)





    












