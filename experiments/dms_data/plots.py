
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HCNN_MODELS_BASE = ['HCNN_biopython_proteinnet',
               'HCNN_biopython_proteinnet_extra_mols',
               'HCNN_pyrosetta_proteinnet_extra_mols']

HCNN_NOISE_LEVELS = ['_0p00', '_0p50']

HCNN_MODELS = [model + noise_level for model in HCNN_MODELS_BASE for noise_level in HCNN_NOISE_LEVELS]

HCNN_MODEL_TO_PRETTY_NAME = {
    'HCNN_biopython_proteinnet_0p00': r'HCNN Biopython no ligands 0.00 $\AA$',
    'HCNN_biopython_proteinnet_0p50': r'HCNN Biopython no ligands 0.50 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p00': r'HCNN Biopython 0.00 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p50': r'HCNN Biopython 0.50 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00': r'HCNN Pyrosetta 0.00 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50': r'HCNN Pyrosetta 0.50 $\AA$'
}

PROTEINMPNN_MODELS = ['proteinmpnn_v_48_002',
                      'proteinmpnn_v_48_020',
                      'proteinmpnn_v_48_030']

PROTEINMPNN_MODEL_TO_PRETTY_NAME = {
    'proteinmpnn_v_48_002': r'ProteinMPNN 0.02 $\AA$',
    'proteinmpnn_v_48_020': r'ProteinMPNN 0.20 $\AA$',
    'proteinmpnn_v_48_030': r'ProteinMPNN 0.30 $\AA$'
}

MODEL_TO_PRETTY_NAME = {
    **HCNN_MODEL_TO_PRETTY_NAME,
    **PROTEINMPNN_MODEL_TO_PRETTY_NAME,
    'ESM-1v (zero shot)': 'ESM-1v (zero shot)'
}


if __name__ == '__main__':

    lw = 1.2
    fontsize = 14
    tick_fontsize = 12
    subplot_size = 4

    ## Compare different HCNN models

    for metric in ['pearsonr', 'spearmanr']:

        df = pd.read_csv(f'df_{metric}.csv')

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
            ax.scatter(df[x], df[y])

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
                ax.scatter(df[x], df[y])

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
        
        plt.tight_layout()
        plot_name = f'comparison_plots_hcnn_{metric}'
        plt.savefig(f'{plot_name}.png')
        plt.savefig(f'{plot_name}.pdf')
        plt.close()
    

    ## Compare different ProteinMPNN models

    for metric in ['pearsonr', 'spearmanr']:

        df = pd.read_csv(f'df_{metric}.csv')

        ncols = len(PROTEINMPNN_MODELS)
        nrows = 1
        colsize = subplot_size
        rowsize = subplot_size
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*colsize, nrows*rowsize), sharex=True, sharey=True)

        ## plot models against each other
        for i, (model1, model2) in enumerate([(model1, model2) for i, model1 in enumerate(PROTEINMPNN_MODELS) for model2 in PROTEINMPNN_MODELS[i+1:] if model1 != model2]):
            ax = axs[i]
            x, y = model1, model2
            ax.scatter(df[x], df[y])

            # lines at zero
            ax.axhline(0, c='k', ls='--', linewidth=lw)
            ax.axvline(0, c='k', ls='--', linewidth=lw)
            
            # set xlim equal to ylim, and plot diagonal line
            xlim = ax.get_xlim()[0], 1
            ylim = ax.get_ylim()[0], 1
            ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
            ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
            ax.plot([0,1],[0,1], c='k', transform=ax.transAxes, linewidth=lw)

            ax.set_xlabel(PROTEINMPNN_MODEL_TO_PRETTY_NAME[x], fontsize=fontsize)
            ax.set_ylabel(PROTEINMPNN_MODEL_TO_PRETTY_NAME[y], fontsize=fontsize)

            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        plt.tight_layout()
        plot_name = f'comparison_plots_proteinmpnn_{metric}'
        plt.savefig(f'{plot_name}.png')
        plt.savefig(f'{plot_name}.pdf')
        plt.close()


    ## Compare ESM-1v (zero shot) to HCNN and ProteinMPNN
    ## Pick the best HCNN and ProteinMPNN models: namely HCNN_pyrosetta_proteinnet_extra_mols_0p50 and proteinmpnn_v_48_030

    model_pairs = [('HCNN_pyrosetta_proteinnet_extra_mols_0p50', 'proteinmpnn_v_48_030'), ('ESM-1v (zero shot)', 'HCNN_pyrosetta_proteinnet_extra_mols_0p50'), ('ESM-1v (zero shot)', 'proteinmpnn_v_48_030')]

    for metric in ['pearsonr', 'spearmanr']:

        df = pd.read_csv(f'df_{metric}.csv')

        ncols = len(model_pairs)
        nrows = 1
        colsize = subplot_size
        rowsize = subplot_size
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*colsize, nrows*rowsize), sharex=True, sharey=True)

        ## plot models against each other
        for i, (model1, model2) in enumerate(model_pairs):
            ax = axs[i]
            x, y = model1, model2
            ax.scatter(df[x], df[y])

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
        
        plt.tight_layout()
        plot_name = f'comparison_plots_esm_and_hcnn_and_proteinmpnn_{metric}'
        plt.savefig(f'{plot_name}.png')
        plt.savefig(f'{plot_name}.pdf')
        plt.close()
