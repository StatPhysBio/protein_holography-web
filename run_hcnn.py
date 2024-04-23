

'''

Change this a bunch:
# 1. Make user select what to return. Whether probabilities, logits, embeddings, or combinations of them. --> DONE
# 2. Make the output be a single .csv file with the res_ids and the requested data. Rows are sites --> DONE BUT NEED TO TEST
3. Make the .txt (multi-pdb) option *optionally* split the output by pdb file, and assign pdbid name to each output file.
4. Use different inference code. Namely, the inference code of TCRpMHC stuff
5. Change model_dir to be the model config name

'''

import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from time import time

from protein_holography_web.inference.hcnn_inference import predict_from_hdf5file, predict_from_pdbfile, load_hcnn_models

import argparse

from protein_holography_web.utils.protein_naming import ind_to_ol_size, ol_to_ind_size

from sklearn.metrics import accuracy_score


def check_input_arguments(args):
    assert args.output_filepath.endswith('.csv'), '--output_filepath must be a ".csv" file.'
    assert args.request, 'At least one of --request must be specified.'
    assert args.hdf5_file or args.pdb_dir, 'Either --hdf5_file or --pdb_dir must be specified.'
    assert not (args.hdf5_file and args.pdb_dir), 'Cannot specify both --hdf5_file and --pdb_dir.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_version', type=str, required=True, choices=['HCNN_0p00', 'HCNN_0p50'],
                        help='Name of HCNN model you want to use. E.g. "HCNN_0p50" is HCNN trained with 0.50 Angstrom noise.')
    
    parser.add_argument('-hf', '--hdf5_file', type=str, default=None,
                        help='Path to an .hdf5 file containing zernikegrams and res_ids to run inference on. Cannot be specified together with --pdb_dir.')
    
    parser.add_argument('-pd', '--pdb_dir', type=str, default=None,
                        help='Directory containing PDB files to run inference on. Inference is run on all sites in the structure. Cannot be specified together with --hdf5_file.')
    
    parser.add_argument('-o', '--output_filepath', type=str, required=True,
                        help='Must be a ".csv file". Embeddings will be saved separately, in a parallel array, with the same filename but with the extension "-embeddings.npz".')
    
    parser.add_argument('-r', '--request', nargs='+', type=str, default='probas', choices=['logprobas', 'probas', 'embeddings', 'logits'],
                        help='Which data to return. Can be a combination of "logprobas", "probas", "embeddings", and "logits".')
    
    parser.add_argument('-bs', '--batch_size', type=int, default=512,
                        help='Batch size for the model.\n'
                             'Will not make a difference if running inference on one PDB at a time (i.e. --pdb_processing is set to "one_at_a_time").')

    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1],
                        help='0 for no, 1 for yes.')

    parser.add_argument('--loading_bar', type=int, default=1, choices=[0, 1],
                        help='0 for no, 1 for yes.')
        
    args = parser.parse_args()


    check_input_arguments(args)

    model_dir_list = os.listdir(os.path.join(os.path.abspath(__file__), 'trained_models', args.model_version))
    models, hparams = load_hcnn_models(model_dir_list)


    if args.hdf5_file is not None:

        print(f'Running inference on zernikegrams in the .hdf5 file: {args.hdf5_file}')
        inference = predict_from_hdf5file()
        print('Accuracy: %.3f' % accuracy_score(inference['targets'], inference['best_indices']))

    elif args.pdb_dir is not None:

        pdb_files = [os.path.join(args.pdb_dir, pdb) for pdb in os.listdir(args.pdb_dir) if pdb.endswith('.pdb')]
        print(f'Running inference on {len(pdb_files)} pdb files found in: {args.pdb_dir}')

        inference = predict_from_pdbfile(pdb_files, models, hparams, args.batch_size, loading_bar=args.loading_bar)

        if len(inference['best_indices'].shape) == 2:
            print('Accuracy of first model in ensemble: %.3f' % accuracy_score(inference['targets'], inference['best_indices'][0, :]))
        else:
            print('Accuracy: %.3f' % accuracy_score(inference['targets'], inference['best_indices']))

    else:
        raise ValueError('Either --hdf5_file or --pdb_dir must be specified.')

    
    ## save res_ids to csv file so they are easily indexable
    ## save embeddings and/or logits in individual numpy arrays
    all_res_ids = inference['res_ids']

    all_res_ids = np.vstack([np.array(res_id.split('_')) for res_id in all_res_ids])
    columns = np.array(['resname', 'pdb', 'chain', 'resnum', 'insertion_code', 'secondary_structure'])
    all_res_ids = np.vstack([columns.reshape(1, -1), all_res_ids])
    all_res_ids = all_res_ids[:, np.array([1, 2, 0, 3, 4])] # rearrange to put pdb in front, and remove secondary structure
    columns = all_res_ids[0]
    all_res_ids = all_res_ids[1:]

    additional_columns = []
    additional_data = []
    for request in args.request:
        if request == 'probas':
            additional_columns.extend([f'proba_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))]) # len(ind_to_ol_size) == num aminoacids
            additional_data.append(inference['probabilities'])
        elif request == 'logprobas':
            additional_columns.extend([f'logproba_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))])
            additional_data.append(np.log(inference['probabilities']))
        elif request == 'logits':
            additional_columns.extend([f'logit_{ind_to_ol_size[i]}' for i in range(len(ind_to_ol_size))])
            additional_data.append(inference['logits'])
    
    columns = np.concatenate([columns, additional_columns])
    data = np.hstack([all_res_ids, np.hstack(additional_data)])
    pd.DataFrame(all_res_ids, columns=columns).to_csv(args.output_filepath, index=False)

    if 'embeddings' in args.request:
        np.savez(args.output_filepath[:-4] + '-embeddings.npz', inference['embeddings'])

    