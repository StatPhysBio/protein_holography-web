

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METADATA_COLUMNS = ['dataset', 'use MT structures', 'measurement type', 'num measurements', 'num structures', 'is single-point', 'is multi-point', 'correlation computation']

MODELS_BASE = ['HCNN_biopython_proteinnet',
               'HCNN_biopython_proteinnet_extra_mols',
               'HCNN_pyrosetta_proteinnet_extra_mols']

NOISE_LEVELS = ['_0p00', '_0p50']

MODELS = [model + noise_level for model in MODELS_BASE for noise_level in NOISE_LEVELS]

MODEL_TO_PRETTY_NAME = {
    'HCNN_biopython_proteinnet_0p00': r'Biopython no ligands 0.00 $\AA$',
    'HCNN_biopython_proteinnet_0p50': r'Biopython no ligands 0.50 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p00': r'Biopython 0.00 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p50': r'Biopython 0.50 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00': r'Pyrosetta 0.00 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50': r'Pyrosetta 0.50 $\AA$'
}


if __name__ == '__main__':

    # make the tables
    os.system('cd Protein_G && python make_results_table.py && cd ..')
    os.system('cd ProTherm/targets && python make_results_table.py && cd ../..')
    os.system('cd S669 && python make_results_table.py && cd ..')
    os.system('cd Ssym && python make_results_table.py && cd ..')
    os.system('cd t4_lysozyme && python make_results_table.py && cd ..')
    os.system('cd skempi && python make_results_table.py && cd ..')
    os.system('cd atlas && python make_results_table.py && cd ..')

    # combine the tables
    table_files = ['Protein_G/protein_g_ddg_experimental-results_table.csv',
                    'ProTherm/targets/protherm_targets_ddg_experimental-results_table.csv',
                    'S669/s669_ddg_experimental-results_table.csv',
                    'Ssym/Ssym_dir/ssym_dir_ddg_experimental-results_table.csv',
                    'Ssym/Ssym_inv/ssym_inv_ddg_experimental-results_table.csv',
                    't4_lysozyme/T4_mutant_ddG_standardized-results_table.csv',
                    'skempi/skempi_v2_cleaned_NO_1KBH-results_table.csv',
                    'atlas/ATLAS_cleaned-results_table.csv']

    dfs = [pd.read_csv(f) for f in table_files]

    df = pd.concat(dfs, ignore_index=True)

    # rearrange columns
    rearranged_columns = ['dataset', 'use MT structures', 'is single-point', 'is multi-point', 'correlation computation']
    rearranged_columns += [f'{model} - Pearsonr' for model in MODELS]
    rearranged_columns += [f'{model} - Spearmanr' for model in MODELS]
    rearranged_columns += [f'{model} - Pearsonr p-value' for model in MODELS]
    rearranged_columns += [f'{model} - Spearmanr p-value' for model in MODELS]
    rearranged_columns += ['measurement type', 'num measurements', 'num structures']

    df = df[rearranged_columns]

    df.to_csv('full_results_table.csv', index=False)


    ## simple comparison plots between columns, though there are rows that are not independent between each other

    lw = 1.2
    fontsize = 12

    for metric in ['Pearsonr', 'Spearmanr']:

        # color by measurement type
        color_list = plt.get_cmap('tab10').colors
        unique_measurement_types = df['measurement type'].unique()
        unique_colors = [color_list[i] for i in range(len(unique_measurement_types))]
        colors = [unique_colors[unique_measurement_types.tolist().index(mt)] for mt in df['measurement type']]

        ncols = len(MODELS_BASE)
        nrows = 1 + len(NOISE_LEVELS)
        colsize = 4
        rowsize = 4
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*colsize, nrows*rowsize), sharex=True, sharey=True)

        # 1) each model and its noisy version
        row = 0
        for col, model_base in enumerate(MODELS_BASE):
            ax = axs[row, col]
            x, y = f'{model_base}{NOISE_LEVELS[0]}', f'{model_base}{NOISE_LEVELS[1]}'
            ax.scatter(df[x + f' - {metric}'], df[y + f' - {metric}'], c=colors)

            # lines at zero
            ax.axhline(0, c='k', ls='--', linewidth=lw)
            ax.axvline(0, c='k', ls='--', linewidth=lw)
            
            # set xlim equal to ylim, and plot diagonal line
            xlim = ax.get_xlim()[0], 1
            ylim = ax.get_ylim()[0], 1
            ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
            ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
            ax.plot([0,1],[0,1], c='k', transform=ax.transAxes, linewidth=lw)

            ax.set_xlabel(MODEL_TO_PRETTY_NAME[x], fontsize=fontsize)
            ax.set_ylabel(MODEL_TO_PRETTY_NAME[y], fontsize=fontsize)

        # 2) and 3) model comparisons, within the same noise level
        for row, noise_level in zip(range(1, len(NOISE_LEVELS) + 1), NOISE_LEVELS):
            model_base_combinations = [(model_base1, model_base2) for i, model_base1 in enumerate(MODELS_BASE) for model_base2 in MODELS_BASE[i+1:] if model_base1 != model_base2]
            for col, (model_base1, model_base2) in enumerate(model_base_combinations):

                ax = axs[row, col]
                x, y = f'{model_base1}{noise_level}', f'{model_base2}{noise_level}'
                ax.scatter(df[x + f' - {metric}'], df[y + f' - {metric}'], c=colors)

                # lines at zero
                ax.axhline(0, c='k', ls='--', linewidth=lw)
                ax.axvline(0, c='k', ls='--', linewidth=lw)
                
                # set xlim equal to ylim, and plot diagonal line
                xlim = ax.get_xlim()[0], 1
                ylim = ax.get_ylim()[0], 1
                ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
                ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
                ax.plot([0,1],[0,1], c='k', transform=ax.transAxes, linewidth=lw)

                ax.set_xlabel(MODEL_TO_PRETTY_NAME[x], fontsize=fontsize)
                ax.set_ylabel(MODEL_TO_PRETTY_NAME[y], fontsize=fontsize)
        
        # place legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=mt, markerfacecolor=unique_colors[i], markersize=10) for i, mt in enumerate(unique_measurement_types)]
        axs[0, 0].legend(handles=legend_handles, loc='upper left')
        
        plt.tight_layout()
        plot_name = f'comparison_plots_{metric}'
        plt.savefig(f'{plot_name}.png')
        plt.savefig(f'{plot_name}.pdf')
        plt.close()

        






