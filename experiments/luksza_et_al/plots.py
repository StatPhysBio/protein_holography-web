
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import argparse

this_file_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True, choices=['HCNN_0p00', 'HCNN_0p50'])
    parser.add_argument('--use_mt_structure', type=int, required=True, choices=[0, 1])
    args = parser.parse_args()

    model_version = args.model_version
    use_mt_structure = args.use_mt_structure

    systems = ['luksza_et_al_tcr1_ec50_sat_mut',
               'luksza_et_al_tcr2_ec50_sat_mut',
               'luksza_et_al_tcr3_ec50_sat_mut',
               'luksza_et_al_tcr4_ec50_sat_mut',
               'luksza_et_al_tcr5_ec50_sat_mut',
               'luksza_et_al_tcr6_ec50_sat_mut',
               'luksza_et_al_tcr7_ec50_sat_mut']
    titles = ['TCR1', 'TCR2', 'TCR3', 'TCR4', 'TCR5', 'TCR6', 'TCR7']
    
    prediction_column = 'log_proba_mt__minus__log_proba_wt'
    target_column = '-EC50'

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

        predictions = df_full[prediction_column].values
        targets = df_full[target_column].values

        plt.figure(figsize=(4, 4))
        plt.scatter(targets, predictions)
        plt.ylabel(r'H-CNN $\Delta \log p$')
        plt.xlabel(r'-$\Delta$EC50')
        plt.title(title)
        correlation, pvalue = spearmanr(predictions, targets)
        plt.text(0.05, 0.88, f'Sr: {correlation:.2f}\np-val: {pvalue:.3f}', transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(os.path.join(this_file_dir, f'{model_version}', 'zero_shot_predictions', f'{system_identifier}-{model_version}-use_mt_structure={use_mt_structure}-scatterplot.png'))
        plt.close()






