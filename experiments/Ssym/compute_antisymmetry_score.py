

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.stats import spearmanr, pearsonr

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True, help='HCNN model name')
    parser.add_argument('--use_mt_structure', type=int, default=0)
    args = parser.parse_args()

    if 'proteinmpnn' in args.model_version:
        model_version_in_filename = 'num_seq_per_target=10'
        pred_column = 'log_p_mt__minus__log_p_wt'
    else:
        model_version_in_filename = args.model_version
        pred_column = 'log_proba_mt__minus__log_proba_wt'
    
    ## assume the two dataframes are parallel, i.e. that each row represents the same site, antisymmetric in each respective dataframe
    dir_df = pd.read_csv(f'Ssym_dir/{args.model_version}/zero_shot_predictions/ssym_dir_ddg_experimental-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}.csv')
    dir_experimental_scores = dir_df['score'].values
    dir_predicted_scores = dir_df[pred_column].values
    dir_pdbids = dir_df['pdbid'].values

    inv_df = pd.read_csv(f'Ssym_inv/{args.model_version}/zero_shot_predictions/ssym_inv_ddg_experimental-{model_version_in_filename}-use_mt_structure={args.use_mt_structure}.csv')
    inv_experimental_scores = inv_df['score'].values
    inv_predicted_scores = inv_df[pred_column].values
    inv_pdbids = inv_df['pdbid'].values

    mask = np.isfinite(dir_experimental_scores) & np.isfinite(dir_predicted_scores) & np.isfinite(inv_experimental_scores) & np.isfinite(inv_predicted_scores)
    dir_experimental_scores = dir_experimental_scores[mask]
    dir_predicted_scores = dir_predicted_scores[mask]
    dir_pdbids = dir_pdbids[mask]
    inv_experimental_scores = inv_experimental_scores[mask]
    inv_predicted_scores = inv_predicted_scores[mask]
    inv_pdbids = inv_pdbids[mask]

    neg_inv_predicted_scores = -inv_predicted_scores
    neg_inv_experimental_scores = -inv_experimental_scores


    ## make scatterplot of dir vs neg_inv predicted scores, color the points by dir_pdbid

    pdbid_to_color = {pdbid: plt.cm.tab20.colors[i] for i, pdbid in enumerate(list(set(dir_pdbids)))}

    # make the plot square

    fig, ax = plt.subplots()
    ax.scatter(dir_predicted_scores, neg_inv_predicted_scores, c=[pdbid_to_color[pdbid] for pdbid in dir_pdbids], alpha=0.5)
    
    # set xlim equal to ylim
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))

    ax.plot([0,1],[0,1], c='k', transform=ax.transAxes)

    ax.axvline(0, c='grey', ls='--')
    ax.axhline(0, c='grey', ls='--')
    ax.set_title(args.model_version + '\n' + 'MAE: %.3f' % np.mean(np.abs(dir_predicted_scores - neg_inv_predicted_scores)))
    ax.set_xlabel('dir predicted scores')
    ax.set_ylabel('neg_inv predicted scores')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/ssym_antisymmetry_score_{args.model_version}-use_mt_structure={args.use_mt_structure}.png')
    plt.close()

