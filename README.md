# protein_holography-web


Notes:
1. The error `{ENVIRONMENT_PATH}/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`, as required by OpenMM, can be fixed via `conda install -c conda-forge libstdcxx-ng`. See https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found
2. For some reason, getting different accuracies on the multi_pdb test? After every run. I do not understand in the slightest. No way the prediction is stochastic. Maybe strucvtural info is stochastic in some way?



