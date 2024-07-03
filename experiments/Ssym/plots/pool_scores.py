
import os
import numpy as np
import pandas as pd

models = []

models += ['HCNN_biopython_proteinnet_0p00',
           'HCNN_biopython_proteinnet_0p50',
           'HCNN_biopython_proteinnet_extra_mols_0p00',
           'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all',
           'HCNN_biopython_proteinnet_extra_mols_0p50', 
           'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all', 
           'HCNN_pyrosetta_proteinnet_extra_mols_0p00', 
           'HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all', 
           'HCNN_pyrosetta_proteinnet_extra_mols_0p50', 
           'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all', 
           'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_invariant_mlp', 
           'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all', 
           'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p00_all']

models += ['proteinmpnn_v_48_002',
           'proteinmpnn_v_48_020',
           'proteinmpnn_v_48_030']

r2_scores = []
for model in models:
    with open(f'ssym_antisymmetry_score_{model}-use_mt_structure=0.txt', 'r') as f:
        antisymmetry_score = float(f.read())
    r2_scores.append(antisymmetry_score)

df = pd.DataFrame({'model': models, 'r2_score_antisymmetry': r2_scores})
df.to_csv('ssym_antisymmetry_scores.csv', index=False)
