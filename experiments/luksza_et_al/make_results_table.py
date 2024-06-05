
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


for tcr_num in [1, 2, 3, 4, 5, 6, 7]:

    dataset_name = f'Luksza et al. - TCR {tcr_num}'
    system_name = f'luksza_et_al_tcr{tcr_num}_ec50_sat_mut'

    correlations_values_in_table = {}

    num_measurements_trace = []

    for hcnn_model in MODELS:

        # load json file with correlations
        with open(f'{hcnn_model}/zero_shot_predictions/{system_name}-{hcnn_model}-use_mt_structure=0-correlations.json', 'r') as f:
            correlations = json.load(f)
        
        pr = correlations['log_proba_mt__minus__log_proba_wt vs. -EC50']['Pr']
        pr_pval = correlations['log_proba_mt__minus__log_proba_wt vs. -EC50']['Pr_pval']

        sr = correlations['log_proba_mt__minus__log_proba_wt vs. -EC50']['Sr']
        sr_pval = correlations['log_proba_mt__minus__log_proba_wt vs. -EC50']['Sr_pval']

        num_measurements = correlations['log_proba_mt__minus__log_proba_wt vs. -EC50']['num']
        
        num_measurements_trace.append(num_measurements)

        correlations_values_in_table[hcnn_model + ' - Pearsonr'] = pr
        correlations_values_in_table[hcnn_model + ' - Pearsonr p-value'] = pr_pval
        correlations_values_in_table[hcnn_model + ' - Spearmanr'] = sr
        correlations_values_in_table[hcnn_model + ' - Spearmanr p-value'] = sr_pval

    if len(set(num_measurements_trace)) > 1:
        print('WARNING: Number of measurements for each model is not the same')

    metadata_values = [dataset_name, '-EC50', num_measurements_trace[-1], 1, True, False, 'per structure']

    metatadata_in_table = dict(zip(METADATA_COLUMNS, metadata_values))

    df = pd.DataFrame(metatadata_in_table, index=[0])
    df = pd.concat([df, pd.DataFrame(correlations_values_in_table, index=[0])], axis=1)

    print(df)

    df.to_csv(f'{system_name}-results_table.csv', index=False)

