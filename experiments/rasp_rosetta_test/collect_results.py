
import os
import numpy as np
import pandas as pd
import json

from protein_holography_web.utils.protein_naming import ol_to_ind_size, ind_to_ol_size

HCNN_MODEL_TO_LATEX_NAME = {
    'HCNN_biopython_proteinnet_extra_mols_0p00': '\HcnnBp',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00': '\HcnnPy',
    'HCNN_biopython_proteinnet_extra_mols_0p50': '\HcnnBpNoise',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50': '\HcnnPyNoise',
    
    'HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all': '\FtHcnnBp',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all': '\FtHcnnPy',
    'HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all': '\FtHcnnBpNoise',
    'HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all': '\FtHcnnPyNoise',
}

HCNN_MODELS_IN_ORDER = list(HCNN_MODEL_TO_LATEX_NAME.keys())


final_dict = {'Model': []}


for model_version in HCNN_MODELS_IN_ORDER:

    with open(f'{model_version}/zero_shot_predictions/test_ddg_rosetta-{model_version}-use_mt_structure=0_correlations.json', 'r') as f:
        data = json.load(f)
    
    final_dict['Model'].append(HCNN_MODEL_TO_LATEX_NAME[model_version])
    
    for pdb in data:
        if pdb == 'overall':
            continue
        
        if pdb not in final_dict:
            final_dict[pdb] = []
        
        final_dict[pdb].append(-data[pdb]['pearson'][0]) # negate it!
    
final_df = pd.DataFrame(final_dict)
final_df.to_csv('rosetta_ddg_results.csv', index=False)
    
    
        

    

