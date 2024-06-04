
import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import pearsonr, spearmanr
import argparse
from protein_holography_web.utils.dms_plots import saturation_mutagenesis_heatmap

this_file_dir = os.path.dirname(os.path.abspath(__file__))


from Bio.Align import substitution_matrices
BLOSUM62 = substitution_matrices.load('BLOSUM62')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True) #, choices=['HCNN_0p00', 'HCNN_0p50'])
    parser.add_argument('--use_mt_structure', type=int, required=True, choices=[0, 1])
    args = parser.parse_args()

    model_version = args.model_version
    use_mt_structure = args.use_mt_structure

    systems = ['hsiue_et_al_H2_sat_mut']
    titles = ['hsiue_et_al_H2_sat_mut']
    
    prediction_column = 'log_proba_mt__minus__log_proba_wt'
    prediction_wt_value = 0.0
    target_column = 'IFN_gamma (pg/ml)'
    target_wt_value = 10596.0


    for system_identifier, title in zip(systems, titles):

        df_full = pd.read_csv(os.path.join(this_file_dir, f'{model_version}', 'zero_shot_predictions', f'{system_identifier}-{model_version}-use_mt_structure={use_mt_structure}.csv'))

        # exlude nans and infs, keep only one wildtype measurement
        is_not_inf_nor_nan_mask = np.logical_and(np.isfinite(df_full[target_column]), np.isfinite(df_full[prediction_column]))
        if 'is_wt' in df_full.columns:
            is_wt_mask = df_full['is_wt']
            first_wt_mask = np.zeros(df_full.shape[0], dtype=bool)
            first_wt_mask[np.arange(df_full.shape[0])[is_wt_mask][0]] = True
            mask = np.logical_and(is_not_inf_nor_nan_mask, np.logical_or(~is_wt_mask, first_wt_mask))
        else:
            print('Warning: no "is_wt" column in csv file, assuming there are no wildtype duplicates.', file=sys.stderr)
            mask = is_not_inf_nor_nan_mask
        
        df_full = df_full.loc[mask].reset_index(drop=True)


        # add blosum scores
        df_full['blosum'] = [BLOSUM62[mutant[0], mutant[-1]] for mutant in df_full['mutant']]

        # make combined score, normalizing hcnn and blosum to be between 0 and 1 and adding them up
        df_full['hcnn_rescaled'] = (df_full[prediction_column] - df_full[prediction_column].min()) / (df_full[prediction_column].max() - df_full[prediction_column].min())
        df_full['blosum_rescaled'] = (df_full['blosum'] - df_full['blosum'].min()) / (df_full['blosum'].max() - df_full['blosum'].min())
        df_full['combined_score'] = df_full['hcnn_rescaled'] + df_full['blosum_rescaled']

        cmap_blosum = {obsv_id: df_full.iloc[obsv_id]['blosum'] for obsv_id in df_full.index}
        sm_blosum = ScalarMappable(norm=Normalize(vmin=min(list(cmap_blosum.values())), vmax=max(list(cmap_blosum.values()))))

        cmap_hcnn = {obsv_id: df_full.iloc[obsv_id]['log_proba_mt__minus__log_proba_wt'] for obsv_id in df_full.index}
        sm_hcnn = ScalarMappable(norm=Normalize(vmin=min(list(cmap_hcnn.values())), vmax=max(list(cmap_hcnn.values()))))

        correlations_dict = {}

        ncols = 5
        nrows = 1
        colsize = 5
        rowsize = 4
        fig, axs = plt.subplots(figsize=(ncols * colsize, nrows * rowsize), ncols=ncols, nrows=nrows)

        ax = axs[0]
        predictions = df_full['log_proba_mt__minus__log_proba_wt'].values
        targets = df_full[target_column].values
        ax.scatter(targets, predictions, color=[sm_blosum.to_rgba(cmap_blosum[obsv_id]) for obsv_id in df_full.index])
        ax.axvline(target_wt_value, color='black', linestyle='--')
        ax.axhline(prediction_wt_value, color='black', linestyle='--')
        ax.set_ylabel(r'H-CNN $\Delta \log p$', fontsize=14)
        ax.set_xlabel(target_column, fontsize=14)
        correlation, pvalue = spearmanr(predictions, targets)
        ax.set_title(f'Color = blosum single mut\nSr: {correlation:.2f}, p-val: {pvalue:.3f}', fontsize=14)
        pr, pr_pval = pearsonr(predictions, targets)
        correlations_dict[f'log_proba_mt__minus__log_proba_wt vs. {target_column}'] = {'Sr': correlation, 'Sr_pval': pvalue, 'Pr': pr, 'Pr_pval': pr_pval, 'num': len(predictions)}

        ax = axs[1]
        predictions = df_full['blosum'].values
        targets = df_full[target_column].values
        ax.scatter(targets, predictions, color=[sm_hcnn.to_rgba(cmap_hcnn[obsv_id]) for obsv_id in df_full.index])
        ax.set_ylabel(r'blosum single mut', fontsize=14)
        ax.set_xlabel(target_column, fontsize=14)
        correlation, pvalue = spearmanr(predictions, targets)
        ax.set_title(f'Color = H-CNN $\Delta \log p$\nSr: {correlation:.2f}, p-val: {pvalue:.3f}', fontsize=14)
        pr, pr_pval = pearsonr(predictions, targets)
        correlations_dict[f'blosum vs. {target_column}'] = {'Sr': correlation, 'Sr_pval': pvalue, 'Pr': pr, 'Pr_pval': pr_pval, 'num': len(predictions)}

        ax = axs[2]
        predictions = df_full['blosum'].values
        targets = df_full['log_proba_mt__minus__log_proba_wt'].values
        ax.scatter(targets, predictions)
        ax.set_ylabel(r'blosum single mut', fontsize=14)
        ax.set_xlabel(r'H-CNN $\Delta \log p$', fontsize=14)
        correlation, pvalue = spearmanr(predictions, targets)
        ax.set_title(f'Sr: {correlation:.2f}, p-val: {pvalue:.3f}', fontsize=14)
        pr, pr_pval = pearsonr(predictions, targets)
        correlations_dict[f'blosum vs. log_proba_mt__minus__log_proba_wt'] = {'Sr': correlation, 'Sr_pval': pvalue, 'Pr': pr, 'Pr_pval': pr_pval, 'num': len(predictions)}

        ax = axs[3]
        predictions = df_full['combined_score'].values
        targets = df_full[target_column].values
        ax.scatter(targets, predictions)
        ax.set_ylabel(r'HCNN $\Delta \log p$ + blosum single mut', fontsize=14)
        ax.set_xlabel(target_column, fontsize=14)
        correlation, pvalue = spearmanr(predictions, targets)
        ax.set_title(f'Sr: {correlation:.2f}, p-val: {pvalue:.3f}', fontsize=14)
        pr, pr_pval = pearsonr(predictions, targets)
        correlations_dict[f'combined_score vs. {target_column}'] = {'Sr': correlation, 'Sr_pval': pvalue, 'Pr': pr, 'Pr_pval': pr_pval, 'num': len(predictions)}

        ax = axs[4]
        weights = np.linspace(0, 1, 100)
        sr_list = []
        for weight in weights:
            combined_score = weight * df_full['blosum_rescaled'] + (1 - weight) * df_full['hcnn_rescaled']
            correlation, pvalue = spearmanr(combined_score, df_full[target_column])
            sr_list.append(correlation)

        ax.plot(weights, sr_list)
        ax.set_xlabel('Weight for blosum single mut', fontsize=14)
        ax.set_ylabel('Spearman correlation', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(this_file_dir, f'{model_version}', 'zero_shot_predictions', f'{system_identifier}-{model_version}-use_mt_structure={use_mt_structure}-scatterplot.png'))
        plt.close()

        # save correlations
        with open(os.path.join(this_file_dir, f'{model_version}', 'zero_shot_predictions', f'{system_identifier}-{model_version}-use_mt_structure={use_mt_structure}-correlations.json'), 'w') as f:
            json.dump(correlations_dict, f, indent=4)
        


        # predictions = df_full[prediction_column].values
        # targets = df_full[target_column].values

        # plt.figure(figsize=(4, 3.5))
        # correlation, pvalue = spearmanr(predictions, targets)
        # plt.scatter(targets, predictions, label=f'Sr: {correlation:.2f}\np-val: {pvalue:.3f}')

        # plt.axvline(target_wt_value, color='black', linestyle='--')
        # plt.axhline(prediction_wt_value, color='black', linestyle='--')

        # plt.ylabel(r'H-CNN $\Delta \log p$', fontsize=12)
        # plt.xlabel(target_column, fontsize=12)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.title(title, fontsize=14)
        # plt.legend(fontsize=10)
        # plt.tight_layout()
        # plt.savefig(os.path.join(this_file_dir, f'{model_version}', 'zero_shot_predictions', f'{system_identifier}-{model_version}-use_mt_structure={use_mt_structure}-scatterplot.png'))
        # plt.close()

        saturation_mutagenesis_heatmap(os.path.join(this_file_dir, f'{model_version}', 'zero_shot_predictions', f'{system_identifier}-{model_version}-use_mt_structure={use_mt_structure}.csv'),
                                       mutant_score_column=prediction_column,
                                       model_type='hcnn')








