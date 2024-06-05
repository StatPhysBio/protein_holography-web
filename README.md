# protein_holography-web


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



## TODO

[] Produce table with all correlation predictions
[] Add usage details in README, showing examples
[X] Better documentation of main scripts (especially argparse of mutation_effect_prediction)
[] Make colab (include it here, so people can use it as a jupyter notebook)

