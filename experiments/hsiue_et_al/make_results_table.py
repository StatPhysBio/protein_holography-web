
import os, sys
import json
import numpy as np
import pandas as pd

sys.path.append('..')
from get_full_table import METADATA_COLUMNS, MODELS


system_name = 'hsiue_et_al_H2_sat_mut'

correlations_values_in_table = {}

num_measurements_trace = []

for hcnn_model in MODELS:

    # load json file with correlations
    with open(f'{hcnn_model}/zero_shot_predictions/{system_name}-{hcnn_model}-use_mt_structure=0-correlations.json', 'r') as f:
        correlations = json.load(f)
    
    pr = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['Pr']
    pr_pval = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['Pr_pval']

    sr = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['Sr']
    sr_pval = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['Sr_pval']

    num_measurements = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['num']
    num_measurements_trace.append(num_measurements)

    correlations_values_in_table[hcnn_model + ' - Pearsonr'] = pr
    correlations_values_in_table[hcnn_model + ' - Pearsonr p-value'] = pr_pval
    correlations_values_in_table[hcnn_model + ' - Spearmanr'] = sr
    correlations_values_in_table[hcnn_model + ' - Spearmanr p-value'] = sr_pval

if len(set(num_measurements_trace)) > 1:
    print('WARNINGL Number of measurements for each model is not the same')

metadata_values = ['Hsiue et al.', False, 'IFN_gamma (pg/ml)', num_measurements_trace[-1], 1, True, False, 'per structure']

metatadata_in_table = dict(zip(METADATA_COLUMNS, metadata_values))

df = pd.DataFrame(metatadata_in_table, index=[0])
df = pd.concat([df, pd.DataFrame(correlations_values_in_table, index=[0])], axis=1)

print(df)

df.to_csv(f'{system_name}-results_table.csv', index=False)





    












