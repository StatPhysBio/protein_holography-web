
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
    parser.add_argument('--system_name', type=str, required=True)
    args = parser.parse_args()


    df_rosetta = pd.read_csv(f'{args.system_name}_ddg_rosetta.csv')
    rosetta_scores_dict = {'-'.join([pdbid, chainid, variant]): score for pdbid, chainid, variant, score in zip(df_rosetta['pdbid'].values, df_rosetta['chainid'].values, df_rosetta['variant'].values, df_rosetta['score'].values)}


    df = pd.read_csv(f'{args.model_version}/zero_shot_predictions/{args.system_name}_ddg_experimental-{args.model_version}-use_mt_structure={args.use_mt_structure}.csv')
    experimental_scores = df['score'].values
    predicted_scores = df['log_proba_mt__minus__log_proba_wt'].values
    pdbids = df['pdbid'].values
    chainids = df['chainid'].values
    variants = df['variant'].values

    mask = np.logical_and(~np.isnan(experimental_scores), ~np.isnan(predicted_scores))
    experimental_scores = experimental_scores[mask]
    predicted_scores = predicted_scores[mask]
    pdbids = pdbids[mask]
    chainids = chainids[mask]
    variants = variants[mask]

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
        if len(pdbid_to_experimental_scores[pdbid]) < 2:
            continue
        curr_experimental_scores = np.array(pdbid_to_experimental_scores[pdbid])
        curr_predicted_scores = np.array(pdbid_to_predicted_scores[pdbid])
        correlations[pdbid] = {
            'pearson': (pearsonr(curr_experimental_scores, curr_predicted_scores)[0], pearsonr(curr_experimental_scores, curr_predicted_scores)[1]),
            'spearman': (spearmanr(curr_experimental_scores, curr_predicted_scores)[0], spearmanr(curr_experimental_scores, curr_predicted_scores)[1]),
            'count': len(curr_experimental_scores)
        }
    
    # add overall correlation
    correlations['overall'] = {
        'pearson': (pearsonr(experimental_scores, predicted_scores)[0], pearsonr(experimental_scores, predicted_scores)[1]),
        'spearman': (spearmanr(experimental_scores, predicted_scores)[0], spearmanr(experimental_scores, predicted_scores)[1]),
        'count': len(experimental_scores)
    }

    # save correlations
    with open(f'{args.model_version}/zero_shot_predictions/{args.system_name}_ddg_experimental-{args.model_version}-use_mt_structure={args.use_mt_structure}_correlations.json', 'w') as f:
        json.dump(correlations, f, indent=4)

