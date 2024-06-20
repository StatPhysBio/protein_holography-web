

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METADATA_COLUMNS = ['dataset', 'use MT structures', 'measurement type', 'num measurements', 'num structures', 'is single-point', 'is multi-point', 'correlation computation']

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
    rearranged_columns += [f'{model} - Pearsonr' for model in HCNN_MODELS + PROTEINMPNN_MODELS]
    rearranged_columns += [f'{model} - Spearmanr' for model in HCNN_MODELS + PROTEINMPNN_MODELS]
    rearranged_columns += [f'{model} - Pearsonr p-value' for model in HCNN_MODELS + PROTEINMPNN_MODELS]
    rearranged_columns += [f'{model} - Spearmanr p-value' for model in HCNN_MODELS + PROTEINMPNN_MODELS]
    rearranged_columns += ['measurement type', 'num measurements', 'num structures']

    df = df[rearranged_columns]

    df.to_csv('full_results_table.csv', index=False)


        






