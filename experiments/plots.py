
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from get_full_table import METADATA_COLUMNS, HCNN_MODELS, PROTEINMPNN_MODELS, HCNN_MODELS_BASE, HCNN_NOISE_LEVELS, HCNN_MODEL_TO_PRETTY_NAME, MODEL_TO_PRETTY_NAME

if __name__ == '__main__':

    df = pd.read_csv('full_results_table.csv')

    ## HCNN: simple comparison plots between columns, though there are rows that are not independent between each other

    lw = 1.2
    fontsize = 14
    tick_fontsize = 12
    subplot_size = 4

    for metric in ['Pearsonr', 'Spearmanr']:

        # color by measurement type
        color_list = plt.get_cmap('tab10').colors
        unique_measurement_types = df['measurement type'].unique()
        unique_colors = [color_list[i] for i in range(len(unique_measurement_types))]
        colors = [unique_colors[unique_measurement_types.tolist().index(mt)] for mt in df['measurement type']]

        ncols = len(HCNN_MODELS_BASE)
        nrows = 1 + len(HCNN_NOISE_LEVELS)
        colsize = subplot_size
        rowsize = subplot_size
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*colsize, nrows*rowsize), sharex=True, sharey=True)

        # 1) each model and its noisy version
        row = 0
        for col, model_base in enumerate(HCNN_MODELS_BASE):
            ax = axs[row, col]
            x, y = f'{model_base}{HCNN_NOISE_LEVELS[0]}', f'{model_base}{HCNN_NOISE_LEVELS[1]}'
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

            ax.set_xlabel(HCNN_MODEL_TO_PRETTY_NAME[x], fontsize=fontsize)
            ax.set_ylabel(HCNN_MODEL_TO_PRETTY_NAME[y], fontsize=fontsize)

            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # 2) and 3) model comparisons, within the same noise level
        for row, noise_level in zip(range(1, len(HCNN_NOISE_LEVELS) + 1), HCNN_NOISE_LEVELS):
            model_base_combinations = [(model_base1, model_base2) for i, model_base1 in enumerate(HCNN_MODELS_BASE) for model_base2 in HCNN_MODELS_BASE[i+1:] if model_base1 != model_base2]
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

                ax.set_xlabel(HCNN_MODEL_TO_PRETTY_NAME[x], fontsize=fontsize)
                ax.set_ylabel(HCNN_MODEL_TO_PRETTY_NAME[y], fontsize=fontsize)
        
        # place legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=mt, markerfacecolor=unique_colors[i], markersize=10) for i, mt in enumerate(unique_measurement_types)]
        axs[0, 0].legend(handles=legend_handles, loc='upper left', fontsize=tick_fontsize)
        
        plt.tight_layout()
        plot_name = f'comparison_plots_hcnn_{metric.lower()}'
        plt.savefig(f'{plot_name}.png')
        plt.savefig(f'{plot_name}.pdf')
        plt.close()
    

    ## ProteinMPNN: simple comparison plots between columns, though there are rows that are not independent between each other

    for metric in ['Pearsonr', 'Spearmanr']:

        # color by measurement type
        color_list = plt.get_cmap('tab10').colors
        unique_measurement_types = df['measurement type'].unique()
        unique_colors = [color_list[i] for i in range(len(unique_measurement_types))]
        colors = [unique_colors[unique_measurement_types.tolist().index(mt)] for mt in df['measurement type']]

        ncols = len(PROTEINMPNN_MODELS)
        nrows = 1
        colsize = subplot_size
        rowsize = subplot_size
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*colsize, nrows*rowsize), sharex=True, sharey=True)

        ## plot models against each other
        for i, (model1, model2) in enumerate([(model1, model2) for i, model1 in enumerate(PROTEINMPNN_MODELS) for model2 in PROTEINMPNN_MODELS[i+1:] if model1 != model2]):

            ax = axs[i]
            x, y = model1, model2
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

            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # place legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=mt, markerfacecolor=unique_colors[i], markersize=10) for i, mt in enumerate(unique_measurement_types)]
        axs[0].legend(handles=legend_handles, loc='upper left', fontsize=tick_fontsize)
        
        plt.tight_layout()
        plot_name = f'comparison_plots_proteinmpnn_{metric.lower()}'
        plt.savefig(f'{plot_name}.png')
        plt.savefig(f'{plot_name}.pdf')
        plt.close()
    

    ## HCNN vs. ProteinMPNN. Use best for each. ProteinMPNN is better without noise for binding but better with noise for stability, so show both

    model_pairs_row_1 = [('HCNN_pyrosetta_proteinnet_extra_mols_0p50', 'proteinmpnn_v_48_002'), ('HCNN_pyrosetta_proteinnet_extra_mols_0p50', 'proteinmpnn_v_48_030')]
    model_pairs_row_2 = [('HCNN_biopython_proteinnet_extra_mols_0p50', 'proteinmpnn_v_48_002'), ('HCNN_biopython_proteinnet_extra_mols_0p50', 'proteinmpnn_v_48_030')]


    for metric in ['Pearsonr', 'Spearmanr']:

        # color by measurement type
        color_list = plt.get_cmap('tab10').colors
        unique_measurement_types = df['measurement type'].unique()
        unique_colors = [color_list[i] for i in range(len(unique_measurement_types))]
        colors = [unique_colors[unique_measurement_types.tolist().index(mt)] for mt in df['measurement type']]

        ncols = max(len(model_pairs_row_1), len(model_pairs_row_2))
        nrows = 2
        colsize = subplot_size
        rowsize = subplot_size
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*colsize, nrows*rowsize), sharex=True, sharey=True)

        ## plot models against each other
        for i, (model1, model2) in enumerate(model_pairs_row_1):

            ax = axs[0, i]
            x, y = model1, model2
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

            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        ## plot models against each other
        for i, (model1, model2) in enumerate(model_pairs_row_2):

            ax = axs[1, i]
            x, y = model1, model2
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

            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # place legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=mt, markerfacecolor=unique_colors[i], markersize=10) for i, mt in enumerate(unique_measurement_types)]
        axs[0, 0].legend(handles=legend_handles, loc='upper left', fontsize=tick_fontsize)
        
        plt.tight_layout()
        plot_name = f'comparison_plots_hcnn_and_proteinmpnn_{metric.lower()}'
        plt.savefig(f'{plot_name}.png')
        plt.savefig(f'{plot_name}.pdf')
        plt.close()
