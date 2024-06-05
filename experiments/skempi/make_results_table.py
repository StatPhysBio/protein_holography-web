
import os
import json
import numpy as np
import pandas as pd


METADATA_COLUMNS = ['dataset', 'measurement type', 'num measurements', 'num structures', 'is single-point', 'is multi-point', 'correlation computation']


MODELS = ['HCNN_biopython_proteinnet_0p00',
          'HCNN_biopython_proteinnet_0p50',
          'HCNN_biopython_proteinnet_extra_mols_0p00',
          'HCNN_biopython_proteinnet_extra_mols_0p50',
          'HCNN_pyrosetta_proteinnet_extra_mols_0p00',
          'HCNN_pyrosetta_proteinnet_extra_mols_0p50']


system_name = 'skempi_v2_cleaned_NO_1KBH'

for curr_comp_in_json, curr_comp in zip(['Per-Structure', 'Overall'],
                                        ['per structure', 'overall']):
    
    for mut_types, is_single, is_multi in zip(['all_types_of_mutations', 'single_point_mutations', 'multi_point_mutations'],
                                              [True, True, False],
                                              [True, False, True]):

    

correlations_values_in_table = {}

num_measurements_trace = []

for hcnn_model in MODELS:

    # load json file with correlations
    with open(f'{hcnn_model}/zero_shot_predictions/{system_name}-{hcnn_model}-use_mt_structure=0-correlations.json', 'r') as f:
        correlations = json.load(f)
    
    pr = correlations['overall']['pearson'][0]
    pr_pval = correlations['overall']['pearson'][1]

    sr = correlations['overall']['spearman'][0]
    sr_pval = correlations['overall']['spearman'][1]

    num_measurements = correlations['overall']['count']
    num_measurements_trace.append(num_measurements)

    correlations_values_in_table[hcnn_model + ' - Pearsonr'] = pr
    correlations_values_in_table[hcnn_model + ' - Pearsonr p-value'] = pr_pval
    correlations_values_in_table[hcnn_model + ' - Spearmanr'] = sr
    correlations_values_in_table[hcnn_model + ' - Spearmanr p-value'] = sr_pval

if len(set(num_measurements_trace)) > 1:
    print('WARNINGL Number of measurements for each model is not the same')

metadata_values = ['Protein G', 'ddG', num_measurements_trace[-1], 1, True, False, 'per structure']

metatadata_in_table = dict(zip(METADATA_COLUMNS, metadata_values))

df = pd.DataFrame(metatadata_in_table, index=[0])
df = pd.concat([df, pd.DataFrame(correlations_values_in_table, index=[0])], axis=1)

print(df)

df.to_csv(f'{system_name}-results_table.csv', index=False)





    












