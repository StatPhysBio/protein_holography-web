
import os
import json
import numpy as np
import pandas as pd


METADATA_COLUMNS = ['measurement type', 'num measurements', 'num structures', 'is single-point', 'is multi-point', 'correlation computation']


MEASUREMENT_TYPE = 'IFN_gamma (pg/ml)'

MODELS = ['HCNN_biopython_proteinnet_0p00',
          'HCNN_biopython_proteinnet_0p50',
          'HCNN_biopython_proteinnet_extra_mols_0p00',
          'HCNN_biopython_proteinnet_extra_mols_0p50',
          'HCNN_pyrosetta_proteinnet_extra_mols_0p00',
          'HCNN_pyrosetta_proteinnet_extra_mols_0p50']


if __name__ == '__main__':

    correlations_values_in_table = {}

    num_measurements_trace = []

    for hcnn_model in MODELS:

        # load json file with correlations
        with open(f'{hcnn_model}/zero_shot_predictions/hsiue_et_al_H2_sat_mut-{hcnn_model}-use_mt_structure=0-correlations.json', 'r') as f:
            correlations = json.load(f)
        
        pr = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['Pr']
        pr_pval = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['Pr_pval']

        sr = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['Sr']
        sr_pval = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['Sr_pval']

        num_measurements = correlations['log_proba_mt__minus__log_proba_wt vs. IFN_gamma (pg/ml)']['num']
        num_measurements_trace.append(num_measurements)

        correlations_values_in_table[hcnn_model + '\nPearsonr'] = pr
        correlations_values_in_table[hcnn_model + '\nPearsonr p-value'] = pr_pval
        correlations_values_in_table[hcnn_model + '\nSpearmanr'] = sr
        correlations_values_in_table[hcnn_model + '\nSpearmanr p-value'] = sr_pval
    
    if len(set(num_measurements_trace)) > 1:
        print('WARNINGL Number of measurements for each model is not the same')
    
    metadata_values = [MEASUREMENT_TYPE, num_measurements_trace[-1]] # TODO continue
    












