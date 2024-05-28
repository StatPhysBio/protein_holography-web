
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

if __name__ == '__main__':

    experimental_scores = pd.read_csv('protein_g_ddg_experimental.csv')['score'].values
    rosetta_scores = pd.read_csv('protein_g_ddg_rosetta.csv')['score'].values

    experimetal_variants = pd.read_csv('protein_g_ddg_experimental.csv')['variant'].values
    rosetta_variants = pd.read_csv('protein_g_ddg_rosetta.csv')['variant'].values

    # match variants, with experimental being a subset
    rosetta_scores_matched = []
    for i, variant in enumerate(experimetal_variants):
        rosetta_scores_matched.append(rosetta_scores[np.where(rosetta_variants == variant)[0][0]])


    print(pearsonr(experimental_scores, rosetta_scores_matched))
    print(spearmanr(experimental_scores, rosetta_scores_matched))
