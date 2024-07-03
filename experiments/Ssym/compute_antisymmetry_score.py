

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

import argparse



HCNN_MODEL_TO_PRETTY_NAME = {
    'HCNN_biopython_proteinnet_0p00': r'HERMES Bp no ligands 0.00',
    'HCNN_biopython_proteinnet_0p50': r'HERMES Bp no ligands 0.50',
    'HCNN_biopython_proteinnet_extra_mols_0p00': r'HERMES Bp 0.00',
    'HCNN_biopython_proteinnet_extra_mols_0p50': r'HERMES Bp 0.50',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00': r'HERMES Py 0.00',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50': r'HERMES Py 0.50',

    'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all': 'HERMES Bp 0.00 + FT',
    'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_invariant_mlp': 'HERMES Bp 0.00 + FT MLP only',
    'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all': 'HERMES Bp 0.50 + FT on 0.00',
    'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all': 'HERMES Bp 0.50 + FT',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all': 'HERMES Py 0.00 + FT',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all': 'HERMES Py 0.50 + FT on 0.00',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all': 'HERMES Py 0.50 + FT',
}

HCNN_MODELS = list(HCNN_MODEL_TO_PRETTY_NAME.keys())

PROTEINMPNN_MODELS = ['proteinmpnn_v_48_002',
                      'proteinmpnn_v_48_020',
                      'proteinmpnn_v_48_030']

PROTEINMPNN_MODEL_TO_PRETTY_NAME = {
    'proteinmpnn_v_48_002': r'ProteinMPNN 0.02',
    'proteinmpnn_v_48_020': r'ProteinMPNN 0.20',
    'proteinmpnn_v_48_030': r'ProteinMPNN 0.30'
}

MODEL_TO_PRETTY_NAME = {
    **HCNN_MODEL_TO_PRETTY_NAME,
    **PROTEINMPNN_MODEL_TO_PRETTY_NAME,
    'ESM-1v (zero shot)': 'ESM-1v (zero shot)'
}

color_list = plt.get_cmap('tab20').colors

blue = color_list[0]
blue_light = color_list[1]
orange = color_list[2]
orange_light = color_list[3]
green = color_list[4]
green_light = color_list[5]
red = color_list[6]
red_light = color_list[7]
purple = color_list[8]
purple_light = color_list[9]
brown = color_list[10]
brown_light = color_list[11]
pink = color_list[12]
pink_light = color_list[13]
gray = color_list[14]
gray_light = color_list[15]
olive = color_list[16]
olive_light = color_list[17]
cyan = color_list[18]
cyan_light = color_list[19]

LATEX_NAME_TO_COLOR = {
    '\\rasp': olive,
    'proteinmpnn_v_48_002': green,
    'proteinmpnn_v_48_020': green_light,
    '\deepsequence': red,
    '\esm': gray,
    'HCNN_biopython_proteinnet_extra_mols_0p00': blue,
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00': purple,
    'HCNN_biopython_proteinnet_extra_mols_0p50': blue_light,
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50': purple_light,
    'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all': cyan,
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all': pink,
    'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all': cyan_light,
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all': pink_light,

    'proteinmpnn_v_48_030': 'black',
    'HCNN_biopython_proteinnet_0p00': 'black',
    'HCNN_biopython_proteinnet_0p50': 'black',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all': 'black',
    'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_invariant_mlp': 'black',
    'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all': 'black',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all': 'black',
}



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
    ax.scatter(dir_predicted_scores, inv_predicted_scores, color=LATEX_NAME_TO_COLOR[args.model_version], alpha=0.5)# , c=[pdbid_to_color[pdbid] for pdbid in dir_pdbids], alpha=0.5)
    
    # set xlim equal to ylim
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    lim_same = min(xlim[0], ylim[1]), max(xlim[1], ylim[0])
    lim_same = min(lim_same[0], -lim_same[1]), max(lim_same[1], -lim_same[0])
    ax.set_xlim(lim_same)
    ax.set_ylim(lim_same)

    ax.plot([0,1],[1,0], c='k', transform=ax.transAxes)

    ax.axvline(0, c='grey', ls='--')
    ax.axhline(0, c='grey', ls='--')
    ax.set_title(MODEL_TO_PRETTY_NAME[args.model_version] + '\n' + r'$R^2$: %.3f' % r2_score(dir_predicted_scores, neg_inv_predicted_scores), fontsize=16)
    ax.set_xlabel('Ssym-direct predictions', fontsize=16)
    ax.set_ylabel('Ssym-reverse predictions', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plot_name = f'plots/ssym_antisymmetry_{args.model_version}-use_mt_structure={args.use_mt_structure}'
    plt.savefig(f'{plot_name}.png')
    plt.savefig(f'{plot_name}.pdf')
    plt.close()

    ## save r2_score to file
    with open(f'plots/ssym_antisymmetry_score_{args.model_version}-use_mt_structure={args.use_mt_structure}.txt', 'w') as f:
        f.write(str(r2_score(dir_predicted_scores, neg_inv_predicted_scores)))

