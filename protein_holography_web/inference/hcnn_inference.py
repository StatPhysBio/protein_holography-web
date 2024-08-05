
import os, sys
import json
from tqdm import tqdm
import torch
import numpy as np
from e3nn import o3
import scipy
import torch

from typing import *

from zernikegrams.preprocessors.pdbs import PDBPreprocessor
import h5py
import hdf5plugin
import tempfile
from rich.progress import Progress

from protein_holography_web.cg_coefficients import get_w3j_coefficients
from protein_holography_web.models import SO3_ConvNet, CGNet

# from protein_holography_web.protein_processing.pipeline import get_zernikegrams_from_pdbfile
from zernikegrams import get_zernikegrams_from_pdbfile
from protein_holography_web.utils.data import ZernikegramsDataset
from protein_holography_web.utils.protein_naming import ol_to_ind_size, ind_to_ol_size


def get_num_components(Lmax, ks, keep_zeros, mode, channels):
    num_components = 0
    if mode == "ns":
        for l in range(Lmax + 1):
            if keep_zeros:
                num_components += (
                    np.count_nonzero(np.array(ks) >= l) * len(channels) * (2 * l + 1)
                )
            else:
                num_components += (
                    np.count_nonzero(
                        np.logical_and(np.array(ks) >= l, (np.array(ks) - l) % 2 == 0)
                    )
                    * len(channels)
                    * (2 * l + 1)
                )

    if mode == "ks":
        for l in range(Lmax + 1):
            num_components += len(ks) * len(channels) * (2 * l + 1)
    
    return num_components


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


def get_zernikegrams_in_parallel(folder_with_pdbs: str,
                                 pdb_files_and_chains: List[Tuple[str, str]],
                                 hparams: Dict,
                                 parallelism: int,
                                 add_same_noise_level_as_training: bool = False):
    
    ## prepare arguments of the pipeline
    channels = get_channels(hparams['channels'])

    get_structural_info_kwargs = {'padded_length': None,
                                    'parser': hparams['parser'],
                                    'SASA': 'SASA' in channels,
                                    'charge': 'charge' in channels,
                                    'DSSP': False,
                                    'angles': False,
                                    'fix': True,
                                    'hydrogens': 'H' in channels,
                                    'extra_molecules': hparams['extra_molecules'],
                                    'multi_struct': 'warn'}

    if add_same_noise_level_as_training:
        add_noise_kwargs = {'noise': hparams['noise'],
                            'noise_seed': hparams['noise_seed']}
    else:
        add_noise_kwargs = None
    
    pdb_list = []
    pdb_to_chain = {}
    for pdbpath, chain in list(pdb_files_and_chains):
        pdb = pdbpath.split('/')[-1][:-4]
        pdb_list.append(pdb)
        pdb_to_chain[pdb] = chain

    def get_residues_only_of_requested_chain(np_protein):
        pdb = np_protein['pdb'].decode()
        chain = pdb_to_chain[pdb]
        res_ids = np.unique(np_protein['res_ids'], axis=0)
        if chain is None: # no specified chain, return all residues
            return res_ids
        else:
            indices = np.where(np.isin(res_ids[:, 2], np.array([chain.encode()])))[0]
            return res_ids[indices]

    get_neighborhoods_kwargs = {'r_max': hparams['rcut'],
                                'remove_central_residue': hparams['remove_central_residue'],
                                'remove_central_sidechain': hparams['remove_central_sidechain'],
                                'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                'get_residues': get_residues_only_of_requested_chain}

    get_zernikegrams_kwargs = {'r_max': hparams['rcut'],
                                'radial_func_mode': hparams['radial_func_mode'],
                                'radial_func_max': hparams['radial_func_max'],
                                'Lmax': hparams['lmax'],
                                'channels': channels,
                                'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                'request_frame': False,
                                'get_physicochemical_info_for_hydrogens': hparams['get_physicochemical_info_for_hydrogens'],
                                'rst_normalization': hparams['rst_normalization']}
    

    processor = PDBPreprocessor(pdb_list, folder_with_pdbs)

    L = np.max([5, processor.pdb_name_length])

    num_components = get_num_components(hparams['lmax'], np.arange(hparams['radial_func_max'] + 1), False, hparams['radial_func_mode'], channels)
    dt = np.dtype(
        [
            ("res_id", f"S{L}", (6,)),
            ("zernikegram", "f4", (num_components,)),
            ("label", "<i4"),
        ]
    )

    hdf5_file = tempfile.NamedTemporaryFile(delete=False)
    hdf5_name = hdf5_file.name

    with h5py.File(hdf5_name, "w") as f:
        f.create_dataset(
            'data',
            shape=(processor.size * 2000,), # assume an average of 2000 atoms per protein to start with, as that is the average single-chain length
            maxshape=(None,),
            dtype=dt,
            chunks=True,
            compression=hdf5plugin.LZ4(),
        )

    with Progress() as bar:
        task = bar.add_task("zernikegrams from multiple pdbs", total=processor.count())
        with h5py.File(hdf5_name, "r+") as f:
            n = 0
            for zgram_data_batch in processor.execute(
                callback=get_zernikegrams_from_pdbfile,
                limit=None,
                params={
                    'get_structural_info_kwargs': get_structural_info_kwargs,
                    'get_neighborhoods_kwargs': get_neighborhoods_kwargs,
                    'get_zernikegrams_kwargs': get_zernikegrams_kwargs,
                    'add_noise_kwargs': add_noise_kwargs
                },
                parallelism=parallelism
            ):
                
                n_added_zgrams = zgram_data_batch['res_id'].shape[0]
                
                if n + n_added_zgrams > f["data"].shape[0]:
                    f["data"].resize((f["data"].shape[0] + n_added_zgrams*3,))
                
                for n_i in range(n_added_zgrams):
                    f["data"][n + n_i] = (zgram_data_batch['res_id'][n_i], zgram_data_batch['zernikegram'][n_i], zgram_data_batch['label'][n_i],)
                
                n += n_added_zgrams
                
                bar.update(task, advance=1)
            
            f["data"].resize((n,))

    return hdf5_name


def predict_from_hdf5file(hdf5_file: str,
                          models: List,
                          hparams: Dict,
                          batch_size: int,
                          regions: Optional[Dict[str, List[Tuple[str, int, str]]]] = None):
    '''
    NOTE: requested chains are already handled in the creation of the hdf5 file, so no need to worry about that here
    '''
    data_irreps, ls_indices = get_data_irreps(hparams)

    with h5py.File(hdf5_file, 'r') as f:
        zgrams_dict = {'res_id': f['data']['res_id'][:], 'zernikegram': f['data']['zernikegram'][:]}

    if regions is None: # return the predictions
        ensemble_predictions_dict = predict_from_zernikegrams(zgrams_dict['zernikegram'], zgrams_dict['res_id'], models, batch_size, data_irreps)
    else: # return the predictions for each region, in a dict indexed by region_name
        ensemble_predictions_dict = {}
        for region_name in regions:
            ensemble_predictions_dict[region_name] = predict_from_zernikegrams(zgrams_dict['zernikegram'], zgrams_dict['res_id'], models, batch_size, data_irreps, region=regions[region_name])
    
    return ensemble_predictions_dict


# @profile
def predict_from_pdbfile(pdb_file: str,
                          models: List,
                          hparams: Dict,
                          batch_size: int,
                          add_same_noise_level_as_training: bool = False,
                          chain: Optional[str] = None,
                          regions: Optional[Dict[str, List[Tuple[str, int, str]]]] = None):

    if chain is not None and regions is not None:
        raise ValueError("Cannot specify both chain and regions")

    data_irreps, ls_indices = get_data_irreps(hparams)

    # this code template would be useful to limit the number of residues to compute zernikegrams - and do inference - for
    if regions is not None:
        def get_residues(np_protein):
            res_ids = np.unique(np_protein['res_ids'], axis=0)
            all_res_ids_info_we_care_about = res_ids[:, 2:5]
            region_ids = []
            for region_name in regions:
                region_ids.extend(regions[region_name])
            region_ids = np.unique(np.array(region_ids).astype(all_res_ids_info_we_care_about.dtype), axis=0)
            indices = np.where(np.isin(all_res_ids_info_we_care_about, region_ids).all(axis=1))[0]
            return res_ids[indices]
    elif chain is not None:
        def get_residues(np_protein):
            res_ids = np.unique(np_protein['res_ids'], axis=0)
            indices = np.where(res_ids[:, 2] == chain.encode())[0]
            return res_ids[indices]
    else:
        get_residues = None

    channels = get_channels(hparams['channels'])

    get_structural_info_kwargs = {'padded_length': None,
                                  'parser': hparams['parser'],
                                  'SASA': 'SASA' in channels,
                                  'charge': 'charge' in channels,
                                  'DSSP': False,
                                  'angles': False,
                                  'fix': True,
                                  'hydrogens': 'H' in channels,
                                  'extra_molecules': hparams['extra_molecules'],
                                  'multi_struct': 'warn'}
    
    if add_same_noise_level_as_training:
        add_noise_kwargs = {'noise': hparams['noise'],
                            'noise_seed': hparams['noise_seed']}
    else:
        add_noise_kwargs = None

    get_neighborhoods_kwargs = {'r_max': hparams['rcut'],
                                'remove_central_residue': hparams['remove_central_residue'],
                                'remove_central_sidechain': hparams['remove_central_sidechain'],
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
    
    zgrams_dict = get_zernikegrams_from_pdbfile(pdb_file, get_structural_info_kwargs, get_neighborhoods_kwargs, get_zernikegrams_kwargs)

    if regions is None: # return the predictions
        ensemble_predictions_dict = predict_from_zernikegrams(zgrams_dict['zernikegram'], zgrams_dict['res_id'], models, batch_size, data_irreps)
    else: # return the predictions for each region, in a dict indexed by region_name
        ensemble_predictions_dict = {}
        for region_name in regions:
            ensemble_predictions_dict[region_name] = predict_from_zernikegrams(zgrams_dict['zernikegram'], zgrams_dict['res_id'], models, batch_size, data_irreps, region=regions[region_name])
    
    return ensemble_predictions_dict


# @profile
def predict_from_zernikegrams(
    np_zgrams: np.ndarray,
    np_res_ids: np.ndarray,
    models: List,
    batch_size: int,
    data_irreps: o3.Irreps,
    region: Optional[List[Tuple[str, int, str]]] = None,
):
    if region is not None:
        region_idxs = get_res_locs_from_tups(np_res_ids, region)
        if len(region_idxs.shape) == 0:
            region_idxs = region_idxs.reshape([1]) # if only one residue, make it still a 1D array instead of a scalar so that stuff doesn't break later
        np_zgrams = np_zgrams[region_idxs]
        np_res_ids = np_res_ids[region_idxs]

    N = np_zgrams.shape[0]
    aas = np_res_ids[:, 0]
    labels = np.array([ol_to_ind_size[x.decode('utf-8')] for x in aas])

    frames = np.zeros((N, 3, 3)) # dummy frames
    dataset = ZernikegramsDataset(np_zgrams, data_irreps, labels, list(zip(list(frames), list(map(tuple, np_res_ids)))))

    ensemble_predictions_dict = {'embeddings': [], 'logits': [], 'probabilities': [], 'best_indices': [], 'targets': None, 'res_ids': np_res_ids}
    for model in models:

        # not sure if I should run re-instantiate the dataloader?
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        curr_model_predictions_dict = model.predict(dataloader, device='cuda' if torch.cuda.is_available() else 'cpu')

        assert (ensemble_predictions_dict['res_ids'][:5, :] == curr_model_predictions_dict['res_ids'].T[:5, :]).all() # sanity check that order of stuff is preserved, have to transpose it for some reason

        ensemble_predictions_dict['embeddings'].append(curr_model_predictions_dict['embeddings'])
        ensemble_predictions_dict['logits'].append(curr_model_predictions_dict['logits'])
        ensemble_predictions_dict['probabilities'].append(curr_model_predictions_dict['probabilities'])
        ensemble_predictions_dict['best_indices'].append(curr_model_predictions_dict['best_indices'])

        if ensemble_predictions_dict['targets'] is None:
            ensemble_predictions_dict['targets'] = curr_model_predictions_dict['targets']
        else:
            assert (ensemble_predictions_dict['targets'][:10] == curr_model_predictions_dict['targets'][:10]).all()

    ensemble_predictions_dict['embeddings'] = np.stack(ensemble_predictions_dict['embeddings'], axis=0) # TODO return concatenated versions of embeddings instead
    ensemble_predictions_dict['logits'] = np.stack(ensemble_predictions_dict['logits'], axis=0)
    ensemble_predictions_dict['probabilities'] = np.stack(ensemble_predictions_dict['probabilities'], axis=0)
    ensemble_predictions_dict['best_indices'] = np.stack(ensemble_predictions_dict['best_indices'], axis=0)

    return ensemble_predictions_dict



def make_string_from_tup(x):
    return (x[0] + str(x[1]) + x[2]).encode()

def get_res_locs_from_tups(
    nh_ids: List,
    loc_tups: List
) -> np.ndarray:
    """Get indices of specific residues based on their residue ids"""
    nh_string_ids = np.array([b''.join(x) for x in nh_ids[:,2:5]])
    loc_string_ids = np.array([make_string_from_tup(x) for x in loc_tups])
    return np.squeeze(np.argwhere(
        np.logical_or.reduce(
            nh_string_ids[None,:] == loc_string_ids[:,None])))



