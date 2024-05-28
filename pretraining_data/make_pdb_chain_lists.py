

import os


if __name__ == '__main__':

    datasets = ['cullpdb_pc30.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains4910',
                'cullpdb_pc50.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains7337',
                'cullpdb_pc70.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains8646',
                'cullpdb_pc90.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains10285']
    
    for dataset in datasets:
        with open(f'./pisces_pdb_lists/{dataset}/{dataset}', 'r') as f:
            lines = f.readlines()[1:]
        pdb_chains = [line[:4]+'_'+line[4] for line in lines]

        sim = int(float(dataset.split('_')[1].split('pc')[1]))
        max_res = dataset.split('_')[2].split('res')[1].split('-')[1].replace('.', 'p')
    
        with open(f'./pisces_pdb_lists/{dataset}/pdb_chain_list_pc={sim}_maxres={max_res}.txt', 'w') as f:
            f.write('\n'.join(pdb_chains))
        

