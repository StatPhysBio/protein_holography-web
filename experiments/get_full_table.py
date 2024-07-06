

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METADATA_COLUMNS = ['dataset', 'use MT structures', 'measurement type', 'num measurements', 'num structures', 'is single-point', 'is multi-point', 'correlation computation']

HCNN_MODELS_BASE = ['HCNN_biopython_proteinnet',
                    'HCNN_biopython_proteinnet_extra_mols',
                    'HCNN_pyrosetta_proteinnet_extra_mols']

HCNN_NOISE_LEVELS = ['_0p00', '_0p50']

# HCNN_MODELS = [model + noise_level for model in HCNN_MODELS_BASE for noise_level in HCNN_NOISE_LEVELS]

HCNN_MODEL_TO_PRETTY_NAME = {
    'HCNN_biopython_proteinnet_0p00': r'HCNN Biopython no ligands 0.00 $\AA$',
    'HCNN_biopython_proteinnet_0p50': r'HCNN Biopython no ligands 0.50 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p00': r'HCNN Biopython 0.00 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p50': r'HCNN Biopython 0.50 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00': r'HCNN Pyrosetta 0.00 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50': r'HCNN Pyrosetta 0.50 $\AA$',

    'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all': 'HCNN Biopython 0.00 $\AA$ \n+ Rosetta ddG All Layers at 0.00 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_invariant_mlp': 'HCNN Biopython 0.00 $\AA$ \n+ Rosetta ddG Invariant MLP at 0.00 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all': 'HCNN Biopython 0.50 $\AA$ \n+ Rosetta ddG All Layers at 0.00 $\AA$',
    'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all': 'HCNN Biopython 0.50 $\AA$ \n+ Rosetta ddG All Layers at 0.50 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all': 'HCNN Pyrosetta 0.00 $\AA$ \n+ Rosetta ddG All Layers at 0.00 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all': 'HCNN Pyrosetta 0.50 $\AA$ \n+ Rosetta ddG All Layers at 0.00 $\AA$',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all': 'HCNN Pyrosetta 0.50 $\AA$ \n+ Rosetta ddG All Layers at 0.50 $\AA$',
}
HCNN_MODELS = list(HCNN_MODEL_TO_PRETTY_NAME.keys())


PROTEINMPNN_MODEL_TO_PRETTY_NAME = {
    'proteinmpnn_v_48_002': r'ProteinMPNN 0.02 $\AA$',
    'proteinmpnn_v_48_020': r'ProteinMPNN 0.20 $\AA$',
    'proteinmpnn_v_48_030': r'ProteinMPNN 0.30 $\AA$'
}

PROTEINMPNN_MODELS = list(PROTEINMPNN_MODEL_TO_PRETTY_NAME.keys())

ESM_MODEL_TO_PRETTY_NAME = {
    'esm_1v_masked_marginals': 'ESM-1v Masked Marginals',
    'esm_1v_wt_marginals': 'ESM-1v Wildtype Marginals'
}

ESM_MODELS = list(ESM_MODEL_TO_PRETTY_NAME.keys())


MODEL_TO_PRETTY_NAME = {
    **HCNN_MODEL_TO_PRETTY_NAME,
    **PROTEINMPNN_MODEL_TO_PRETTY_NAME,
    **ESM_MODEL_TO_PRETTY_NAME,
    'ESM-1v (zero shot)': 'ESM-1v (zero shot)'
}


if __name__ == '__main__':

    # make the tables
    os.system('cd Protein_G && python make_results_table.py && cd ..')
    os.system('cd VAMP && python make_results_table.py && cd ..')
    os.system('cd ProTherm/targets && python make_results_table.py && cd ../..')
    os.system('cd S669 && python make_results_table.py && cd ..')
    os.system('cd Ssym && python make_results_table.py && cd ..')
    os.system('cd t4_lysozyme && python make_results_table.py && cd ..')
    os.system('cd skempi && python make_results_table.py && cd ..')
    os.system('cd atlas && python make_results_table.py && cd ..')

    # combine the tables
    table_files = ['Protein_G/protein_g_ddg_experimental-results_table.csv',
                    'VAMP/vamp_ddg_experimental-results_table.csv',
                    'ProTherm/targets/protherm_targets_ddg_experimental-results_table.csv',
                    'S669/s669_ddg_experimental-results_table.csv',
                    'Ssym/Ssym_dir/ssym_dir_ddg_experimental-results_table.csv',
                    'Ssym/Ssym_inv/ssym_inv_ddg_experimental-results_table.csv',
                    't4_lysozyme/T4_mutant_ddG_standardized-results_table.csv',
                    'skempi/skempi_v2_cleaned_NO_1KBH-results_table.csv',
                    'atlas/ATLAS_cleaned-results_table.csv']

    dfs = [pd.read_csv(f) for f in table_files]

    df = pd.concat(dfs, ignore_index=True)

    df.to_csv('full_results_table.csv', index=False)


        






