
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_VALID_PDBIDS = 10 # from RaSP

SEED = 42
np.random.seed(SEED)

def fermi_transform(ddg: np.ndarray, beta: float = 0.4, alpha: float = 3.0) -> np.ndarray:
    return 1 / (1 + np.exp(-beta*(ddg - alpha)))

if __name__ == '__main__':

    train_and_valid_data = pd.read_csv('train_and_valid_ddg.csv')
    train_and_valid_pdbids = train_and_valid_data['pdbid'].unique()

    # split out a validation set
    np.random.shuffle(train_and_valid_pdbids)
    valid_pdbids = train_and_valid_pdbids[:NUM_VALID_PDBIDS]
    train_pdbids = train_and_valid_pdbids[NUM_VALID_PDBIDS:]

    train_data = train_and_valid_data[train_and_valid_data['pdbid'].isin(train_pdbids)]
    valid_data = train_and_valid_data[train_and_valid_data['pdbid'].isin(valid_pdbids)]
    test_data = pd.read_csv('test_ddg.csv')

    # plot a distribution of score values in two subplots, before and after the fermi transform
    scores = list(train_data['score'].values) + list(valid_data['score'].values) + list(test_data['score'].values)
    scores = np.array(scores)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].hist(scores, bins=50)
    axs[0].set_title('Before Fermi Transform', fontsize=16)
    axs[1].hist(fermi_transform(scores), bins=50)
    axs[1].set_title('After Fermi Transform', fontsize=16)
    # plot fermi funtion
    x = np.linspace(-20, 80, 1000)
    y = fermi_transform(x)
    axs[2].plot(x, y)
    axs[2].axvline(-1, c='black', ls='--')
    axs[2].axvline(7, c='black', ls='--')
    axs[2].set_title('Fermi Transform', fontsize=16)
    plt.tight_layout()
    plt.savefig('score_distributions.png')
    plt.close()

    # apply fermi transform to scores
    train_data['score'] = fermi_transform(train_data['score'].values)
    valid_data['score'] = fermi_transform(valid_data['score'].values)
    test_data['score'] = fermi_transform(test_data['score'].values)

    # save the data
    train_data.to_csv('train_targets.csv', index=False)
    valid_data.to_csv('valid_targets.csv', index=False)
    test_data.to_csv('test_targets.csv', index=False)




