

import os
import numpy as np
import pandas as pd


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
