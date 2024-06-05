
import os
from glob import glob
import numpy as np
import pandas as pd
import json
from scipy.stats import spearmanr, pearsonr

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True, help='HCNN model name')
    parser.add_argument('--use_mt_structure', type=int, default=0)
    args = parser.parse_args()


    df = pd.read_csv(f'{args.model_version}/zero_shot_predictions/T4_mutant_ddG_standardized-{args.model_version}-use_mt_structure={args.use_mt_structure}.csv')
    experimental_scores = df['ddG'].values
    predicted_scores = df['log_proba_mt__minus__log_proba_wt'].values
    pdbids = df['wt_pdb'].values

    mask = np.logical_and(~np.isnan(experimental_scores), ~np.isnan(predicted_scores))
    experimental_scores = experimental_scores[mask]
    predicted_scores = predicted_scores[mask]
    pdbids = pdbids[mask]

    # split by pdbid
    pdbid_to_experimental_scores = {}
    pdbid_to_predicted_scores = {}
    for pdbid, experimental_score, predicted_score in zip(pdbids, experimental_scores, predicted_scores):
        if pdbid not in pdbid_to_experimental_scores:
            pdbid_to_experimental_scores[pdbid] = []
            pdbid_to_predicted_scores[pdbid] = []
        pdbid_to_experimental_scores[pdbid].append(experimental_score)
        pdbid_to_predicted_scores[pdbid].append(predicted_score)
    
    # calculate correlations
    correlations = {}
    for pdbid in pdbid_to_experimental_scores:
        experimental_scores = np.array(pdbid_to_experimental_scores[pdbid])
        predicted_scores = np.array(pdbid_to_predicted_scores[pdbid])
        correlations[pdbid] = {
            'pearson': (pearsonr(experimental_scores, predicted_scores)[0], pearsonr(experimental_scores, predicted_scores)[1]),
            'spearman': (spearmanr(experimental_scores, predicted_scores)[0], spearmanr(experimental_scores, predicted_scores)[1])
        }
    
    # save correlations
    with open(f'{args.model_version}/zero_shot_predictions/T4_mutant_ddG_standardized-{args.model_version}-use_mt_structure={args.use_mt_structure}_correlations.json', 'w') as f:
        json.dump(correlations, f, indent=4)





