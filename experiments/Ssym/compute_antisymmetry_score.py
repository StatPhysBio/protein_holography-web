

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


    ## assume the two dataframes are parallel, i.e. that each row represents the same site, antisymmetric in each respective dataframe

    dir_df = pd.read_csv(f'Ssym_dir/{args.model_version}/zero_shot_predictions/ssym_dir_ddg_experimental-{args.model_version}-use_mt_structure={args.use_mt_structure}.csv')
    dir_experimental_scores = dir_df['score'].values
    dir_predicted_scores = dir_df['log_proba_mt__minus__log_proba_wt'].values
    dir_pdbids = dir_df['pdbid'].values

    inv_df = pd.read_csv(f'Ssym_inv/{args.model_version}/zero_shot_predictions/ssym_inv_ddg_experimental-{args.model_version}-use_mt_structure={args.use_mt_structure}.csv')
    inv_experimental_scores = inv_df['score'].values
    inv_predicted_scores = inv_df['log_proba_mt__minus__log_proba_wt'].values
    inv_pdbids = inv_df['pdbid'].values

    neg_inv_predicted_scores = -inv_predicted_scores


    ## make scatterplot of dir vs neg_inv predicted scores, color the points by dir_pdbid

    fig, ax = plt.subplots()
    ax.plot(ax.get_xlim(), ax.get_ylim(), c='black', ls='-')
    ax.axvline(0, c='grey', ls='--')
    ax.axhline(0, c='grey', ls='--')
    ax.scatter(dir_predicted_scores, neg_inv_predicted_scores, c=dir_pdbids, cmap='tab20', alpha=0.5)
    ax.set_title(args.model_version + '\n' + 'MAE: %.3f' % np.mean(np.abs(dir_predicted_scores - neg_inv_predicted_scores)))
    ax.set_xlabel('dir predicted scores')
    ax.set_ylabel('neg_inv predicted scores')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/ssym_antisymmetry_score_{args.model_version}-use_mt_structure={args.use_mt_structure}.png')
    plt.close()

