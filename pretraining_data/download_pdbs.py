


import os
import urllib.request
import pandas as pd
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/gscratch/stf/gvisan01/pisces_pdbs/')
    args = parser.parse_args()

    datasets = ['cullpdb_pc30.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains4910',
                'cullpdb_pc50.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains7337',
                'cullpdb_pc70.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains8646',
                'cullpdb_pc90.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains10285']
    
    pdbs_to_download = []
    for dataset in datasets:
        with open(f'./pisces_pdb_lists/{dataset}/{dataset}', 'r') as f:
            lines = f.readlines()[1:]
        pdbs = [line[:4] for line in lines]
        pdbs_to_download.extend(pdbs)
    pdbs_to_download = list(set(pdbs_to_download))

    os.makedirs(args.output_dir, exist_ok=True)

    for pdb in tqdm(pdbs_to_download):

        # skip if the file already exists
        if os.path.exists(os.path.join(args.output_dir, f'{pdb}.pdb')):
            continue

        try:
            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb}.pdb', os.path.join(args.output_dir, f'{pdb}.pdb'))
        except Exception as e:
            print(f'Error downloading {pdb}: {e}')
            continue
    

    
