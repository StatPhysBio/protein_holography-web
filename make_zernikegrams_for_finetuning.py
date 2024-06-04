

import os, sys
import numpy as np
import gzip, pickle
import json
from tqdm import tqdm
from protein_holography_web.inference.hcnn_inference import get_channels
from zernikegrams import get_zernikegrams_from_pdbfile

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_version', type=str, required=True,
                        help='HCNN model name. The script will take the hparams from this model and use them to preprocess the protein structures accoridng to the way the model was pre-trained.')
    
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='The directory containing the pdb files to process. The script will process all pdbs in this directory.')
    
    parser.add_argument('--pdb_list', type=str, default=None,
                        help='A .txt file containing a list of pdbids to process. Each line should contain a pdbid. If not specified, the script will process all pdbs in the pdb_dir.')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The directory to save the zernikegrams to. The zernikegrams will be saved in a file named after the model version.')
    
    args = parser.parse_args()

    ## get list of pdbs
    if args.pdb_list is not None:
        with open(args.pdb_list, 'r') as f:
            pdbs = f.read().splitlines()
    else:
        print('No pdb list provided, processing all pdbs in the pdb_dir...')
        pdbs = [pdb[:-4] for pdb in os.listdir(args.pdb_dir) if pdb.endswith('.pdb')]
    
    ## get hparams, assume they are the same for all models in the model version (the relevant haprams should be!!!)
    trained_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', args.model_version)
    model_dir_list = [os.path.join(trained_models_path, model_rel_path) for model_rel_path in os.listdir(trained_models_path)]
    with open(os.path.join(model_dir_list[0], 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    

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

    get_neighborhoods_kwargs = {'r_max': hparams['rcut'],
                                'remove_central_residue': hparams['remove_central_residue'],
                                'remove_central_sidechain': hparams['remove_central_sidechain'],
                                'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                                'get_residues': None}

    get_zernikegrams_kwargs = {'r_max': hparams['rcut'],
                               'radial_func_mode': hparams['radial_func_mode'],
                               'radial_func_max': hparams['radial_func_max'],
                               'Lmax': hparams['lmax'],
                               'channels': channels,
                               'backbone_only': set(hparams['channels']) == set(['CA', 'C', 'O', 'N']),
                               'request_frame': False,
                               'get_physicochemical_info_for_hydrogens': hparams['get_physicochemical_info_for_hydrogens'],
                               'rst_normalization': hparams['rst_normalization']}
    
    ## run pipeline!

    all_amino_acids = []
    all_pdbids = []
    all_chainids = []
    all_resnums = []
    all_zernikegrams = []

    for pdb in tqdm(pdbs):
        pdb_path = os.path.join(args.pdb_dir, pdb + '.pdb')
        data = get_zernikegrams_from_pdbfile(pdb_path, get_structural_info_kwargs, get_neighborhoods_kwargs, get_zernikegrams_kwargs)
        
        res_ids = data['res_id']
        zernikegrams = data['zernikegram']

        # unpack the res_id to make it more readable/user-friendly

        amino_acids = np.array([res_id[0].decode('utf-8') for res_id in res_ids])
        pdbids =  np.array([res_id[1].decode('utf-8') for res_id in res_ids])
        chainids = np.array([res_id[2].decode('utf-8') for res_id in res_ids])
        resnums = np.array([int(res_id[3].decode('utf-8')) for res_id in res_ids])

        all_amino_acids.append(amino_acids)
        all_pdbids.append(pdbids)
        all_chainids.append(chainids)
        all_resnums.append(resnums)
        all_zernikegrams.append(zernikegrams)
    
    all_amino_acids = np.concatenate(all_amino_acids)
    all_pdbids = np.concatenate(all_pdbids)
    all_chainids = np.concatenate(all_chainids)
    all_resnums = np.concatenate(all_resnums)
    all_zernikegrams = np.vstack(all_zernikegrams)

    num_zernikegrams = all_zernikegrams.shape[0]
    

    ## make output_dir if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    ## save the zernikegrams
    with gzip.open(os.path.join(args.output_dir, args.model_version + '.npz.gz'), 'wb') as f:
        pickle.dump({'pdbid': all_pdbids, 'chainid': all_chainids, 'resnum': all_resnums, 'amino_acid': all_amino_acids, 'zernikegram': all_zernikegrams}, f)
    
    print(f"Saved {num_zernikegrams} zernikegrams for {len(pdbs)} pdb files in {os.path.join(args.output_dir, args.model_version + '.npz.gz')}")






