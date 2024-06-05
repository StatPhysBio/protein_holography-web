
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

    df_rosetta = pd.read_csv(f'{args.system_name}/{args.system_name.lower()}_ddg_rosetta.csv')
    rosetta_scores_dict = {'-'.join([pdbid, chainid, variant]): score for pdbid, chainid, variant, score in zip(df_rosetta['pdbid'].values, df_rosetta['chainid'].values, df_rosetta['variant'].values, df_rosetta['score'].values)}

    df = pd.read_csv(f'{args.system_name}/{args.model_version}/zero_shot_predictions/{args.system_name.lower()}_ddg_experimental-{args.model_version}-use_mt_structure={args.use_mt_structure}.csv')
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
    
    ## repeat, but filter for rows that have a rosetta score in the range [-1 ; 7]

    filtered_pdbids = []
    filtered_experimental_scores = []
    filtered_predicted_scores = []
    for pdbid, chainid, variant, experimental_score, predicted_score in zip(pdbids, chainids, variants, experimental_scores, predicted_scores):
        if rosetta_scores_dict['-'.join([pdbid, chainid, variant])] >= -1 and rosetta_scores_dict['-'.join([pdbid, chainid, variant])] <= 7:
            filtered_pdbids.append(pdbid)
            filtered_experimental_scores.append(experimental_score)
            filtered_predicted_scores.append(predicted_score)
    filtered_pdbids = np.array(filtered_pdbids)
    filtered_experimental_scores = np.array(filtered_experimental_scores)
    filtered_predicted_scores = np.array(filtered_predicted_scores)

    # split by pdbid
    pdbid_to_filtered_experimental_scores = {}
    pdbid_to_filtered_predicted_scores = {}
    for pdbid, experimental_score, predicted_score in zip(filtered_pdbids, filtered_experimental_scores, filtered_predicted_scores):
        if pdbid not in pdbid_to_filtered_experimental_scores:
            pdbid_to_filtered_experimental_scores[pdbid] = []
            pdbid_to_filtered_predicted_scores[pdbid] = []
        pdbid_to_filtered_experimental_scores[pdbid].append(experimental_score)
        pdbid_to_filtered_predicted_scores[pdbid].append(predicted_score)
    
    # calculate correlations
    for pdbid in pdbid_to_filtered_experimental_scores:
        if len(pdbid_to_filtered_experimental_scores[pdbid]) < 2:
            continue
        curr_filtered_experimental_scores = np.array(pdbid_to_filtered_experimental_scores[pdbid])
        curr_filtered_predicted_scores = np.array(pdbid_to_filtered_predicted_scores[pdbid])
        correlations[pdbid] = {
            **correlations[pdbid],
            **{'pearson - filtered rosetta [-1;7]': (pearsonr(curr_filtered_experimental_scores, curr_filtered_predicted_scores)[0], pearsonr(curr_filtered_experimental_scores, curr_filtered_predicted_scores)[1]),
               'spearman - filtered rosetta [-1;7]': (spearmanr(curr_filtered_experimental_scores, curr_filtered_predicted_scores)[0], spearmanr(curr_filtered_experimental_scores, curr_filtered_predicted_scores)[1]),
               'count - filtered rosetta [-1;7]': len(curr_filtered_experimental_scores)}
        }
    
    # add overall correlation
    if len(filtered_experimental_scores) >= 2:
        correlations['overall'] = {
            **correlations['overall'],
            **{'pearson - filtered rosetta [-1;7]': (pearsonr(filtered_experimental_scores, filtered_predicted_scores)[0], pearsonr(filtered_experimental_scores, filtered_predicted_scores)[1]),
               'spearman - filtered rosetta [-1;7]': (spearmanr(filtered_experimental_scores, filtered_predicted_scores)[0], spearmanr(filtered_experimental_scores, filtered_predicted_scores)[1]),
               'count - filtered rosetta [-1;7]': len(filtered_experimental_scores)}
        }
    
    # save correlations
    with open(f'{args.system_name}/{args.model_version}/zero_shot_predictions/{args.system_name.lower()}_ddg_experimental-{args.model_version}-use_mt_structure={args.use_mt_structure}_correlations.json', 'w') as f:
        json.dump(correlations, f, indent=4)





