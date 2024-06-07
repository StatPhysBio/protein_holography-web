# protein_holography-web


## Colab

You can run predictions easily on the [![Google Colab Notebook](https://colab.research.google.com/drive/1JQxkXeGZJYYcPNglN3rYUYiOuUOkwJPL#scrollTo=pfOyfkx_tvBf)].


## Installation

**Step 1:** Install the required packages


**Step 2:** Clone and install `zernikegrams` repo, which is used to process protein structures.
```bash
git clone git@github.com:StatPhysBio/zernikegrams.git
cd zernikegrams
pip install .
```
This clones the reository and installs it using pip, making it usable as a package.
We will soon make this package available on PyPI for easier download.

**(Optional) Step 3:** Install pyrosetta. This is required for the use of models trained on structures processed using pyrosetta. A license is available at no cost to academics and can be obtained [here](https://www.pyrosetta.org/home/licensing-pyrosetta). We are aware of the limitations posed by pyrosetta's license and are working on releasing a version that uss biopython instead and other open source code soon.

To download pyrosetta, after obtaining a license from the link above, follow instructions [here](https://www.pyrosetta.org/downloads#h.6vttn15ac69d). We recommend downloading the .whl file and installing with pip. Our models were trained and evaluated using version `PyRosetta4.Release.python39.linux.release-335`. We do not foresee issues with using alternative versions, but cannot guarantee compatibility at this time.

**Step 4:** Finally, install `protein_holography_web` as a package.
```bash
pip install .
```

Feel free to use `pip install -e .` instead if you plan on making changes to the code and want to test them without reinstalling the package.


Notes:
1. The error `{ENVIRONMENT_PATH}/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`, as required by OpenMM, can be fixed via `conda install -c conda-forge libstdcxx-ng`. See https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found



## Usage

### Models

We provide the following pre-trained models:
- `HCNN_biopython_proteinnet_0p00`: trained on ~10k CASP12 ProteinNet chains, using the preprocessing pipeline from the RaSP paper.
- `HCNN_biopython_proteinnet_extra_mols_0p00`: trained on ~10k CASP12 ProteinNet chains, using the preprocessing pipeline from the RaSP paper, *except we keep extra ligands and ions (not water)*.
- `HCNN_pyrosetta_proteinnet_extra_mols_0p00`: trained on ~10k CASP12 ProteinNet chains, using *pyrosetta* to preprocess protein structures as described in the HCNN paper. This procedure includes ligands and ions, excludes water, and - constrary to RaSP - does **not** substitute non-canonical amino-acids.

We also provide models trained on the same data, except adding gaussian noise with standard deviation of 0.5 Angstrom to the coordinates. To use these mdoels, just replace `_0p00` with `_0p50` in the model name.

The relative performance of the models on zero-shot pediction of stability, binding, and immune activity measurements can be found below under **Benchmarking results**.

Note that, to use the pyrosetta models, a local installation of pyrosetta is necessary.


### Getting site-level mutation probabilities and embeddings for all sites in PDB files

The script `run_hcnn_on_pdbfiles.py` can be given as input a set of PDB files - with optionally pdb-specific chains - and it will output a csv file where every row is a uniquely-identified site, and columns are the site's mutation probabilities. If embeddings are requested, they will be outputted in a separate file called `{CSV_FILENAME}-embeddings.npy`.

Some common use cases:

1. Get all probabilities and embeddings for all sites in the PDB files found in `pdbs`.
```bash
python run_hcnn_on_pdbfiles.py -pd pdbs --m HCNN_biopython_proteinnet_0p00 -o all_sites.csv -r probas embeddings
```
The above command will output two csv files: one called `all_sites.csv` with mutation probabilities, the other called `all_sites-embeddings.npy` with embeddings, for all sites in the PDB files found in the `pdbs` directory.

2. Request to process specific pdbs and chains.
The requested pdbs and chains should be listed in a text file, with one pdb and chain per line. For example, assume the file `my_pdbs_and_chains.txt` contains the following:
```
1ao7 A
1qrn A
1qrn B
```
Then, to process only these pdbs and chains, run:
```bash
python run_hcnn_on_pdbfiles.py -pd pdbs -pn my_pdbs_and_chains.txt --m HCNN_biopython_proteinnet_0p00 -o specific_chains.csv -r probas embeddings
```

**NOTE:** If a requested pdb file is not found in the directory, the script will automatically attempt to download it from the RCSB website.


Please run `python run_hcnn_on_pdbfiles.py -h` for more information.


### Scoring specific mutations

Sometimes, it is useful to score specific mutations. The script `zero_shot_mutation_effect_prediction_with_hcnn.py` can be used for this purpose. It takes as input a csv file with columns corresponding to: the mutation, the chain, and the pdb file of the wildtype. The script will output a csv file with the mutation probabilities and embeddings for the requested mutations.

If desired, the script supports the use of the mutant structure to predict the mutation effect. This can be done by providing the mutant pdb file in the csv file in the appropriate column.

The columns are not expected to have specific names, but the names must ben provided as input to the script.

Run `python zero_shot_mutation_effect_prediction_with_hcnn.py -h` for more information on the script, and see `experiments/Protein_G/` for a simple example.


## Benchmarking results




## TODO

[] Produce table with all correlation predictions \\
[X] Add usage details in README, showing examples \\
[X] Better documentation of main scripts (especially argparse of mutation_effect_prediction) \\
[] Make colab (include it here, so people can use it as a jupyter notebook) \\

