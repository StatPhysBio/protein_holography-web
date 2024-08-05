

import os
import numpy as np
import pandas as pd
import time


def test_run_hcnn_on_pdbfiles__directory_only():

    outfilename = "test_output_directory_only"

    os.system(f"python -u ../run_hcnn_on_pdbfiles.py \
                                -pd pdbs \
                                -o {outfilename}.csv \
                                -m HCNN_biopython_proteinnet_0p50 \
                                -r probas embeddings \
                                -v 1")
    
    df = pd.read_csv(f"{outfilename}.csv")
    embeddings = np.load(f"{outfilename}-embeddings.npy")

    assert df.shape[0] == embeddings.shape[0]

    os.system(f"rm -r {outfilename}.csv")
    os.system(f"rm -r {outfilename}-embeddings.npy")


def test_run_hcnn_on_pdbfiles__some_pdbids_specified():

    pdbs_and_chains_file = "pdbs_and_chains.txt"
    pdbs = ['1ao7', '1qrn']
    with open(pdbs_and_chains_file, "w") as f:
        f.write("\n".join(pdbs))

    outfilename = "test_some_pdbids_specified"

    os.system(f"python -u ../run_hcnn_on_pdbfiles.py \
                                -pd pdbs \
                                -pn {pdbs_and_chains_file} \
                                -o {outfilename}.csv \
                                -m HCNN_biopython_proteinnet_0p50 \
                                -r probas embeddings \
                                -v 1")
    
    df = pd.read_csv(f"{outfilename}.csv")
    embeddings = np.load(f"{outfilename}-embeddings.npy")

    assert df.shape[0] == embeddings.shape[0]

    assert sorted(df['pdb'].unique()) == sorted(pdbs)

    os.remove(pdbs_and_chains_file)
    os.system(f"rm -r {outfilename}.csv")
    os.system(f"rm -r {outfilename}-embeddings.npy")


def test_run_hcnn_on_pdbfiles__some_pdbids_and_chains_specified():

    pdbs_and_chains_file = "pdbs_and_chains.txt"
    pdbs = ['1ao7', '1qrn']
    chains = ['A', 'A']
    with open(pdbs_and_chains_file, "w") as f:
        f.write("\n".join([' '.join([pdb, chain]) for pdb, chain in zip(pdbs, chains)]))

    outfilename = "test_some_pdbids_and_chains_specified"

    os.system(f"python -u ../run_hcnn_on_pdbfiles.py \
                                -pd pdbs \
                                -pn {pdbs_and_chains_file} \
                                -o {outfilename}.csv \
                                -m HCNN_biopython_proteinnet_0p50 \
                                -r probas embeddings \
                                -v 1")
    
    df = pd.read_csv(f"{outfilename}.csv")
    embeddings = np.load(f"{outfilename}-embeddings.npy")

    assert df.shape[0] == embeddings.shape[0]

    assert sorted(df['pdb'].unique()) == sorted(pdbs)

    for pdb, chain in zip(pdbs, chains):
        assert (df[df['pdb'] == pdb]['chain'].unique() == [chain]).all()

    os.remove(pdbs_and_chains_file)
    os.system(f"rm -r {outfilename}.csv")
    os.system(f"rm -r {outfilename}-embeddings.npy")


def test_run_hcnn_on_pdbfiles__downloading_pdbs():

    pdbs_and_chains_file = "pdbs_and_chain_to_download.txt"
    pdbs = ['1ao7', '1qrn']
    with open(pdbs_and_chains_file, "w") as f:
        f.write("\n".join(pdbs))

    outfilename = "test_download_pdbs"

    os.system(f"python -u ../run_hcnn_on_pdbfiles.py \
                                -pd pdbs_downloaded \
                                -pn {pdbs_and_chains_file} \
                                -o {outfilename}.csv \
                                -m HCNN_biopython_proteinnet_0p50 \
                                -r probas embeddings \
                                -v 1")
    
    df = pd.read_csv(f"{outfilename}.csv")
    embeddings = np.load(f"{outfilename}-embeddings.npy")

    assert df.shape[0] == embeddings.shape[0]

    assert sorted(df['pdb'].unique()) == sorted(pdbs)

    os.remove(pdbs_and_chains_file)
    os.system(f"rm -r pdbs_downloaded")
    os.system(f"rm -r {outfilename}.csv")
    os.system(f"rm -r {outfilename}-embeddings.npy")


def test_run_hcnn_on_pdbfiles__with_parallelism_and_without():

    parallelism_list = [1, 2, 3]

    for parallelism in parallelism_list:

        outfilename_parallel = f"parallel_{parallelism}"

        start = time.time()
        os.system(f"python -u ../run_hcnn_on_pdbfiles.py \
                                    -pd pdbs \
                                    -pp {parallelism} \
                                    -o {outfilename_parallel}.csv \
                                    -m HCNN_biopython_proteinnet_extra_mols_0p00 \
                                    -r logprobas embeddings \
                                    -v 0")
        print(f"Elapsed time with requested parallelism {parallelism} (need to also have the necessary cores!): {time.time() - start} seconds")
        
        df_parallel = pd.read_csv(f"{outfilename_parallel}.csv")
        embeddings_parallel = np.load(f"{outfilename_parallel}-embeddings.npy")

        assert df_parallel.shape[0] == embeddings_parallel.shape[0]


    outfilename_series = "series"

    start = time.time()
    os.system(f"python -u ../run_hcnn_on_pdbfiles.py \
                                -pd pdbs \
                                -o {outfilename_series}.csv \
                                -m HCNN_biopython_proteinnet_extra_mols_0p00 \
                                -r logprobas embeddings \
                                -v 0")
    print(f"Elapsed time in series: {time.time() - start} seconds")
    
    df_series = pd.read_csv(f"{outfilename_series}.csv")
    embeddings_series = np.load(f"{outfilename_series}-embeddings.npy")

    assert df_series.shape[0] == embeddings_series.shape[0]

    assert df_parallel.shape[0] == df_series.shape[0]


if __name__ == '__main__':

    test_run_hcnn_on_pdbfiles__with_parallelism_and_without()