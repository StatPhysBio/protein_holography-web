# HERMES: Holographic Equivariant neuRal network model for Mutational Effect and Stability prediction

![Schematic of HERMES](hermes.pdf)


## Running on Colab

You can run predictions easily on the [Google Colab Notebook](https://colab.research.google.com/drive/1JQxkXeGZJYYcPNglN3rYUYiOuUOkwJPL).


## Installing and running locally

NOTE: Currently, there is a conflict between `openmm` and `pytorch`, whereby it's challenging to install both with CUDA support. We are currently struggling to replicate the installation on our local HPC cluster - which we once managed to do (sigh...). Install `pytorch` with CPU support only, however, gives no issues. Try installing `pytorch` with CUDA first, and if that doesn't work, install it with CPU only. We are working on a foolproof solution - if you have any leads please let us know by posting an Issue.

**Step 1:** Create environment and install pytorch.
```bash
conda create -n protholo python=3.9
conda activate protholo
```

Install `pytorch==1.13.1` with or without CUDA depending on whether you have a GPU available, following https://pytorch.org/get-started/previous-versions/. For example:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia # with cuda for gpu support
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch # cpu only, use this if having issues with step 2
```

**Step 2:** Install `zernikegrams` package, which we use for protein preprocessing.
```bash
conda install zernikegrams -c statphysbio -c conda-forge
```
TD;DR: if you are experiencing issues with the installation and you have `pytorch` with CUDA installed, install `pytorch` with CPU only.
Installing `zernikegrams` will also install other necessary packages such as `openmm`. As outlined above, in our environment we are experiencing a conflict between `openmm` and `pytorch` that we are currently working on resolving, and cannot guarantee GPU support at this time (though it works without issues on colab). \\



**(Optional) Step 3:** Install pyrosetta. This is required for the use of models trained on structures processed using pyrosetta. A license is available at no cost to academics and can be obtained [here](https://www.pyrosetta.org/home/licensing-pyrosetta). We are aware of the limitations posed by pyrosetta's license and are working on releasing a version that uss biopython instead and other open source code soon.

To download pyrosetta, after obtaining a license from the link above, follow instructions [here](https://www.pyrosetta.org/downloads#h.6vttn15ac69d). We recommend downloading the .whl file and installing with pip. Our models were trained and evaluated using version `PyRosetta4.Release.python39.linux.release-335`. We do not foresee issues with using alternative versions, but cannot guarantee compatibility at this time.


**Step 4:** Install `protein_holography_web` as a package. This will install some of the other necessary packages as well.
```bash
pip install .
```


Feel free to use `pip install -e .` instead if you plan on making changes to the code and want to test them without reinstalling the package.


Installation tips:
1. The error `{ENVIRONMENT_PATH}/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`, as required by OpenMM, can be fixed via `conda install -c conda-forge libstdcxx-ng`. See https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found
2. Somehow can't install pdbfixer AFTER conda



## Provided pre-trained and fine-tuned models

We are in the process of re-naming models in the repository. For now, please **use the old names**.\\
[new name] <--> [old name]

We provide the following pre-trained models:
- `HERMES_Bp_000` <--> `HCNN_biopython_proteinnet_extra_mols_0p00`: trained on ~10k CASP12 ProteinNet chains, using the preprocessing pipeline from the RaSP paper based on Biopython, *except we keep extra ligands and ions (not water)*.
- `HERMES_Bp_050` <--> `HCNN_biopython_proteinnet_extra_mols_0p50`: same as above, but trained on data with added gaussian noise with standard deviation of 0.5 Angstrom.
- `HERMES_Py_000` <--> `HCNN_pyrosetta_proteinnet_extra_mols_0p00`: trained on ~10k CASP12 ProteinNet chains, using *pyrosetta* to preprocess protein structures as described in the HCNN paper. This procedure includes ligands and ions, excludes water, and - constrary to RaSP - does **not** substitute non-canonical amino-acids.
- `HERMES_Py_050` <--> `HCNN_pyrosetta_proteinnet_extra_mols_0p50`: same as above, but trained on data with added gaussian noise with standard deviation of 0.5 Angstrom.

We also provide the same models fine-tuned on stability ddG values computed with Rosetta:
- `HERMES_Bp_000_FT_Ros_ddG` <--> `HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all`
- `HERMES_Bp_050_FT_Ros_ddG` <--> `HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all`
- `HERMES_Py_000_FT_Ros_ddG` <--> `HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all`
- `HERMES_Py_050_FT_Ros_ddG` <--> `HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all`

Please see the paper for a comprehensive benchmarking evaluation. The tl;dr is: fine-tuned models work better, and all four perform similarly. Pre-trained-only models work better with 0.50 Angstrom noise, and with pyrosetta preprocedding.

Note that, to use the pyrosetta models, a local installation of pyrosetta is necessary.


## Getting site-level mutation probabilities and embeddings for all sites in PDB files

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


## Scoring specific mutations

Sometimes, it is useful to score specific mutations. The script `zero_shot_mutation_effect_prediction_with_hcnn.py` can be used for this purpose. It takes as input a csv file with columns corresponding to: the mutation, the chain, and the pdb file of the wildtype. The script will output a csv file with the mutation probabilities and embeddings for the requested mutations.

If desired, the script supports the use of the mutant structure to predict the mutation effect. This can be done by providing the mutant pdb file in the csv file in the appropriate column.

The columns are not expected to have specific names, but the names must ben provided as input to the script.

Run `python zero_shot_mutation_effect_prediction_with_hcnn.py -h` for more information on the script, and see `experiments/Protein_G/` for a simple example.


## Want to fine-tune on your mutation effect dataset?

Fine-tuning can be easily done in three steps.

1. **Prepare the data.** Prepare the targets in three .csv files, which must have `{train}`, `{valid}`, and `{test}` in the name. Each .csv file must have the following columns: `[pdbid, chainid, variant, score]`. Also, place all the pdbfiles for training, validation and testing in a single directory.

2. **Generate inputs (aka zernikegrams or holograms).** For faster training, we pre-generate the inputs and store them in a .npz file. Run `make_zernikegrams_for_finetuning.py` to generate the inputs, providing as arguments, the model you want to make inputs for, the directory of pdbfiles, whether to add noise to structures, and the output directory.

3. **Fine-tune the model.** Run `finetune_hcnn.py` to fine-tune the model. You need to provide a config file with the necessary information, see `/training_data/finetuning/rosetta_ddg/configs/HCNN_biopython_proteinnet_extra_mols_0p00__all.yaml` for a thorough example.











