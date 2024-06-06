

import os
import numpy as np
import pandas as pd

METADATA_COLUMNS = ['dataset', 'use MT structures', 'measurement type', 'num measurements', 'num structures', 'is single-point', 'is multi-point', 'correlation computation']

MODELS = ['HCNN_biopython_proteinnet_0p00',
          'HCNN_biopython_proteinnet_0p50',
          'HCNN_biopython_proteinnet_extra_mols_0p00',
          'HCNN_biopython_proteinnet_extra_mols_0p50',
          'HCNN_pyrosetta_proteinnet_extra_mols_0p00',
          'HCNN_pyrosetta_proteinnet_extra_mols_0p50']


if __name__ == '__main__':

    # make the tables
    os.system('cd Protein_G && python make_results_table.py && cd ..')
    os.system('cd ProTherm/targets && python make_results_table.py && cd ../..')
    os.system('cd S669 && python make_results_table.py && cd ..')
    os.system('cd Ssym && python make_results_table.py && cd ..')
    os.system('cd t4_lysozyme && python make_results_table.py && cd ..')
    os.system('cd skempi && python make_results_table.py && cd ..')
    os.system('cd atlas && python make_results_table.py && cd ..')
    os.system('cd hsiue_et_al && python make_results_table.py && cd ..')
    os.system('cd luksza_et_al && python make_results_table.py && cd ..')

    # combine the tables
    table_files = ['Protein_G/protein_g_ddg_experimental-results_table.csv',
                'ProTherm/targets/protherm_targets_ddg_experimental-results_table.csv',
                'S669/s669_ddg_experimental-results_table.csv',
                'Ssym/Ssym_dir/ssym_dir_ddg_experimental-results_table.csv',
                'Ssym/Ssym_inv/ssym_inv_ddg_experimental-results_table.csv',
                't4_lysozyme/T4_mutant_ddG_standardized-results_table.csv',
                'skempi/skempi_v2_cleaned_NO_1KBH-results_table.csv',
                'atlas/ATLAS_cleaned-results_table.csv',
                'hsiue_et_al/hsiue_et_al_H2_sat_mut-results_table.csv',
                'luksza_et_al/luksza_et_al_tcr1_ec50_sat_mut-results_table.csv',
                'luksza_et_al/luksza_et_al_tcr2_ec50_sat_mut-results_table.csv',
                'luksza_et_al/luksza_et_al_tcr3_ec50_sat_mut-results_table.csv',
                'luksza_et_al/luksza_et_al_tcr4_ec50_sat_mut-results_table.csv',
                'luksza_et_al/luksza_et_al_tcr5_ec50_sat_mut-results_table.csv',
                'luksza_et_al/luksza_et_al_tcr6_ec50_sat_mut-results_table.csv',
                'luksza_et_al/luksza_et_al_tcr7_ec50_sat_mut-results_table.csv',]

    dfs = [pd.read_csv(f) for f in table_files]

    df = pd.concat(dfs, ignore_index=True)

    df.to_csv('full_results_table.csv', index=False)


