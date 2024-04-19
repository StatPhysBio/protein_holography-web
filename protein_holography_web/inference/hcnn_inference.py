
# from protein_holography_pytorch.utils.hcnn.prediction_parser import ...

import os, sys

import json
import torch
import numpy as np

from typing import *

from protein_holography_pytorch.preprocessing_faster import get_zernikegrams_from_structures

from protein_holography_pytorch.utils.hcnn.prediction_parser import get_region_metrics, average_region_metrics


def get_channels(channels_str):

    if channels_str == 'dlpacker':
        channels = ['C', 'N', 'O', 'S', "all other elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all other AAs"]
    elif channels_str == 'dlpacker_plus':
        channels = ['CAlpha', 'C', 'N', 'O', 'S', "all other elements", 'charge',
                          b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                          b'S', b'T', b'W', b'Y', b'V', b'G',
                         "all other AAs"]
    elif channels_str == 'AAs':
        channels = [b'A', b'R', b'N', b'D', b'C', b'Q', b'E', b'H', b'I', b'L', b'K', b'M', b'F', b'P', 
                    b'S', b'T', b'W', b'Y', b'V', b'G', "all other AAs"]
    else:
        channels = channels_str.split(',')
    
    return channels


def get_data_irreps(hparams):
    import numpy as np
    from e3nn import o3

    channels = get_channels(hparams['channels'])
    
    # construct data irreps from hparams
    mul_by_l = []
    if hparams['radial_func_mode'] == 'ks':
        for l in range(hparams['lmax'] + 1):
            mul_by_l.append((hparams['radial_func_max']+1) * len(channels))
    
    elif hparams['radial_func_mode'] == 'ns':
        ns = np.arange(hparams['radial_func_max'] + 1)
        for l in range(hparams['lmax'] + 1):
            # assuming not keeping zeros... because why would we?
            mul_by_l.append(np.count_nonzero(np.logical_and(np.array(ns) >= l, (np.array(ns) - l) % 2 == 0)) * len(channels))

    data_irreps = o3.Irreps('+'.join([f'{mul}x{l}e' for l, mul in enumerate(mul_by_l)]))
    ls_indices = np.concatenate([[l]*(2*l+1) for l in data_irreps.ls])

    return data_irreps, ls_indices


def load_hcnn_models(model_dirs: List[str]):

    '''
    Assume that all models have the same hparams and same data_irreps
    '''

    from protein_holography_pytorch.cg_coefficients import get_w3j_coefficients
    from protein_holography_pytorch.models import SO3_ConvNet, CGNet

    
    models = []
    for model_dir in model_dirs:

        # get hparams from json
        with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
            hparams = json.load(f)
        
        data_irreps, ls_indices = get_data_irreps(hparams)
        
        if hparams['normalize_input']:
            normalize_input_at_runtime = True
        else:
            normalize_input_at_runtime = False

        # setup device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running on %s.' % (device))

        # load w3j coefficients
        w3j_matrices = get_w3j_coefficients()
        for key in w3j_matrices:
            # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
            if device is not None:
                w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
            else:
                w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
            w3j_matrices[key].requires_grad = False
        
        if hparams['model_type'] == 'so3_convnet':
            model = SO3_ConvNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=normalize_input_at_runtime).to(device)
        elif hparams['model_type'] == 'cgnet':
            model = CGNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=normalize_input_at_runtime).to(device)
        else:
            raise ValueError(f"Unknown model type {hparams['model_type']}")
        model.load_state_dict(torch.load(os.path.join(model_dir, 'lowest_valid_loss_model.pt'), map_location=torch.device(device)))
        model.eval()
        models.append(model)
    
    return models, hparams # assume that all models have the same hparams and same data_irreps


# def predict_from_hdf5(


def predict_from_pdbfiles(pdb_files: List[str],
                          models,
                          hparams: List[Dict],
                          max_atoms: int = 100000): # mnight have to change max_atoms to make sure all possible atoms are included
    '''
    '''

    data_irreps, ls_indices = get_data_irreps(hparams)

    ## this code template would be useful to limit the number of residues to compute zernikegrams - and do inference - for
    # if compute_zgrams_only_for_requested_regions:
    #     def get_residues(np_protein):
    #         res_ids = np.unique(np_protein['res_ids'], axis=0)
    #         all_res_ids_info_we_care_about = res_ids[:, 2:5]
    #         region_ids = []
    #         for region_name in regions:
    #             region_ids.extend(regions[region_name])
    #         region_ids = np.unique(np.array(region_ids).astype(all_res_ids_info_we_care_about.dtype), axis=0)
    #         indices = np.where(np.isin(all_res_ids_info_we_care_about, region_ids).all(axis=1))[0]
    #         return res_ids[indices]
    # else:
    #     get_residues = None
    get_residues = None


    get_neighborhoods_kwargs = {'r_max': hparams['rcut'],
                                'remove_central_residue': hparams['remove_central_residue'],
                                'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                'get_residues': get_residues}


    get_zernikegrams_kwargs = {'r_max': hparams['rcut'],
                               'radial_func_mode': hparams['radial_func_mode'],
                               'radial_func_max': hparams['radial_func_max'],
                               'Lmax': hparams['lmax'],
                               'channels': get_channels(hparams['channels']),
                               'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                               'request_frame': False,
                               'get_physicochemical_info_for_hydrogens': hparams['get_physicochemical_info_for_hydrogens'],
                               'rst_normalization': hparams['rst_normalization']}

    zgrams_dict = get_zernikegrams_from_structures(pose, get_neighborhoods_kwargs, get_zernikegrams_kwargs, max_atoms)


    # predict_from_zernikegrams()



def predict_from_zernikegrams(
    zernikegrams: np.ndarray,
    models,
    hparams: List[Dict],
    data_irreps,
    ls_indices
):
    '''
    '''

    # TODO continue this
    for model in models:
        pass

        # model.predict()

        # predict(self,
        #         dataloader: torch.utils.data.DataLoader,
        #         emb_i: int = -1,
        #         device: str = 'cpu',
        #         verbose: bool = False,
        #         loading_bar: bool = False) -> Dict:

        # return {
        #     'embeddings': embeddings_all,
        #     'logits': y_hat_all_logits,
        #     'probabilities': softmax(y_hat_all_logits, axis=-1),
        #     'best_indices': y_hat_all_index,
        #     'targets': y_all,
        #     'res_ids': res_ids_all
        # }


