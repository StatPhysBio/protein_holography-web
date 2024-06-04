

import os
import numpy as np

np.random.seed(42)

PERC_TRAIN = 0.90
PERC_VAL = PERC_TEST = (1.0 - PERC_TRAIN) / 2

MAX_NUM_PROTEINS_PER_TRAINING_FILE = 800


if __name__ == '__main__':

    datasets = ['cullpdb_pc30.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains4910',
                'cullpdb_pc50.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains7337',
                'cullpdb_pc70.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains8646',
                'cullpdb_pc90.0_res0.0-2.5_noBrks_noDsdr_len40-10000_R0.25_Xray+EM_d2024_05_21_chains10285']
    
    for dataset in datasets:
        with open(f'./pisces_pdb_lists/{dataset}/{dataset}', 'r') as f:
            lines = f.readlines()[1:]
        pdbs = [line[:4] for line in lines]

        sim = int(float(dataset.split('_')[1].split('pc')[1]))
        max_res = dataset.split('_')[2].split('res')[1].split('-')[1].replace('.', 'p')

        np.random.shuffle(pdbs)

        num_train = int(PERC_TRAIN * len(pdbs))
        num_val = int(PERC_VAL * len(pdbs))
        num_test = len(pdbs) - num_train - num_val

        print(dataset)
        print(f'Number of pdbs: {len(pdbs)}')
        print(f'Number of pdbs in training set: {num_train}')
        print(f'Number of pdbs in validation set: {num_val}')
        print(f'Number of pdbs in testing set: {num_test}')
        print()

        train_pdbs = pdbs[:num_train]
        val_pdbs = pdbs[num_train:num_train+num_val]
        test_pdbs = pdbs[num_train+num_val:]

        # split the training set into multiple files
        num_train_files = int(np.ceil(num_train / MAX_NUM_PROTEINS_PER_TRAINING_FILE))
        for i in range(num_train_files):
            with open(f'./pisces_pdb_lists/{dataset}/pdb_list_training__{i}_pc={sim}_maxres={max_res}.txt', 'w') as f:
                f.write('\n'.join(train_pdbs[i*MAX_NUM_PROTEINS_PER_TRAINING_FILE:(i+1)*MAX_NUM_PROTEINS_PER_TRAINING_FILE]))

        with open(f'./pisces_pdb_lists/{dataset}/pdb_list_validation_pc={sim}_maxres={max_res}.txt', 'w') as f:
            f.write('\n'.join(val_pdbs))
        
        with open(f'./pisces_pdb_lists/{dataset}/pdb_list_testing_pc={sim}_maxres={max_res}.txt', 'w') as f:
            f.write('\n'.join(test_pdbs))
        

