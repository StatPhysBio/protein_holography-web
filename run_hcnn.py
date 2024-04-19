

'''

Change this a bunch:
1. Make user select what to return. Whether probabilities, ligits, embeddings, or combinations of them.
2. Make the output be a single .csv file with the res_ids and the requested data. Rows are sites
3. Make the .txt (multi-pdb) option *optionally* split the output by pdb file, and assign pdbid name to each output file.
4. Use different inference code. Namely, the inference code 

'''

import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from time import time

from runtime.hcnn_aa_classifier.src import hcnn_aa_classifier_inference

import argparse
from protein_holography_pytorch.utils.argparse import *

from sklearn.metrics import accuracy_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', nargs='+', type=str, required=True,
                        help='Directory(ies) containing the trained model(s).')
    
    parser.add_argument('-i', '--input_filepath', type=str, required=True,
                        help='Either a .hdf5 filepath, a .pdb filepath, or a .txt filepath containing a list of PDBs that are to be found in --pdb_dir.')
    
    parser.add_argument('-o', '--output_filepath_no_extension', type=str, required=True,
                        help='Output filename without any extension.\n'
                             'The embeddings will be saved to this file with the extension ".npz". The res_ids will be saved to this file with the extension ".csv".')
    
    parser.add_argument('-r', '--request', type=str, default='both', choices=['embeddings', 'logits', 'both'],
                        help='Whether to save embeddings, logits, or both.')
    
    parser.add_argument('-bs', '--batch_size', type=int, default=512,
                        help='Batch size for the model.\n'
                             'Will not make a difference if running inference on one PDB at a time (i.e. --pdb_processing is set to "one_at_a_time").')

    parser.add_argument('--verbose', type=int, default=0, help='0 for no, 1 for yes.')
    parser.add_argument('--loading_bar', type=int, default=1, help='0 for no, 1 for yes.')

    ##### for using pdb file(s)
    parser.add_argument('--pdb_dir', type=optional_str, default=None,
                        help='Only relevant if --input_filepath is a .txt file, in which case it must be specified.\n'
                             'Directory containing PDB files.')
    
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Only relevant if --input_filepath is a .txt file.\n'
                             'Num workers (alias: parallelism) for multiprocessing of PDBs.\n')
    
    parser.add_argument('--parser', type=str, default='pyrosetta',
                        choices = ['pyrosetta', 'biopython'],
                        help='Only relevant if --input_filepath is a .pdb or a .txt file.\n'
                              'Which parser to use for parsing PDB files.')
    ###################################################################
    
    
    args = parser.parse_args()

    if args.input_filepath.endswith('.hdf5'):
        
        print(f'Running inference on zernikegrams in the .hdf5 file: {args.input_filepath}')

        
        inference = hcnn_aa_classifier_inference(args.model_dir,
                                                    output_filepath=None, # ensures that the embeddings are not saved to disk, but instead returned
                                                    data_filepath=args.input_filepath,
                                                    verbose=args.verbose,
                                                    loading_bar=args.loading_bar,
                                                    batch_size=args.batch_size)

        print('Accuracy: %.3f' % accuracy_score(inference['targets'], inference['best_indices']))

    elif args.input_filepath.endswith('.pdb'):
        print(f'Running inference on a single PDB: {args.input_filepath}')

        inference = hcnn_aa_classifier_inference(args.model_dir,
                                    output_filepath=None, # ensures that the embeddings are not saved to disk, but instead returned
                                    data_filepath=args.input_filepath,
                                    parser=args.parser,
                                    verbose=args.verbose,
                                    loading_bar=args.loading_bar,
                                    batch_size=args.batch_size)

        if len(inference['best_indices'].shape) == 2:
            print('Accuracy of first model in ensemble: %.3f' % accuracy_score(inference['targets'], inference['best_indices'][:, 0]))
        else:
            print('Accuracy: %.3f' % accuracy_score(inference['targets'], inference['best_indices']))

    elif args.input_filepath.endswith('.txt'):
        print(f'Running inference on the list of PDBs found in {args.input_filepath}')

        start = time()
        inference = hcnn_aa_classifier_inference(args.model_dir,
                                                    data_filepath=args.input_filepath,
                                                    pdb_dir=args.pdb_dir,
                                                    parser=args.parser,
                                                    verbose=args.verbose,
                                                    loading_bar=args.loading_bar,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers)

        print('Took %d seconds.' % (time() - start))

        print('Accuracy: %.3f' % accuracy_score(inference['targets'], inference['best_indices']))
    
    else:
        raise ValueError('--input_filepath must be either a .pdb filepath or a .txt filepath.')

    
    ## save res_ids to csv file so they are easily indexable
    ## save embeddings and/or logits in individual numpy arrays
    all_res_ids = inference['res_ids']

    all_res_ids = np.vstack([np.array(res_id.split('_')) for res_id in all_res_ids])
    columns = np.array(['resname', 'pdb', 'chain', 'resnum', 'insertion_code', 'secondary_structure'])
    all_res_ids = np.vstack([columns.reshape(1, -1), all_res_ids])
    all_res_ids = all_res_ids[:, np.array([1, 2, 0, 3, 4])] # rearrange to put pdb in front, and remove secondary structure
    columns = all_res_ids[0]
    all_res_ids = all_res_ids[1:]

    pd.DataFrame(all_res_ids, columns=columns).to_csv(args.output_filepath_no_extension + '-res_ids.csv', index=False)

    if args.request in {'logits', 'both'}:
        np.savez(args.output_filepath_no_extension + '-logits.npz', inference['logits'])
    
    if args.request in {'embeddings', 'both'}:
        np.savez(args.output_filepath_no_extension + '-embeddings.npz', inference['embeddings'])
    