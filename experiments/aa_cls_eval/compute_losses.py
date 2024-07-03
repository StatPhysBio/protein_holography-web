

import os
import numpy as np
import pandas as pd

import torch
from sklearn.metrics import accuracy_score

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




if __name__ == '__main__':

    final_dict = {'Model': [], 'CE Loss': [], 'Accuracy': []}

    for model_version in HCNN_MODELS_IN_ORDER:

        df = pd.read_csv(f'output/{model_version}.csv')

        true_indices = []
        pred_logits = []
        for i, row in df.iterrows():
            true_indices.append(ol_to_ind_size[row['resname']])

            curr_logits = []
            for i in range(20):
                aa = ind_to_ol_size[i]
                curr_logits.append(row[f'logit_{aa}'])
            pred_logits.append(np.array(curr_logits))
        
        true_indices = np.array(true_indices)
        pred_logits = np.array(pred_logits)

        celoss = torch.nn.CrossEntropyLoss()(torch.tensor(pred_logits), torch.tensor(true_indices))
        accuracy = accuracy_score(true_indices, np.argmax(pred_logits, axis=1))

        final_dict['Model'].append(HCNN_MODEL_TO_LATEX_NAME[model_version])
        final_dict['CE Loss'].append(celoss.item())
        final_dict['Accuracy'].append(accuracy)
    
    final_df = pd.DataFrame(final_dict)
    final_df.to_csv('aa_cls_results.csv', index=False)



    

