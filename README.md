# protein_holography-web

TODO:
1. Add `requirements.txt` and general tips on installation procedure
2. Profile and speed up inference. Currently takes 20/30 seconds per protein, which is a little too long.
3. Add quick running of T4-Lysozymes and SKEMPI (at least a subset) ddG prediction to check zero-shot performance
4. T4-Lysozymes is done --> good performance of un-noised model
5. SKEMPI is done --> good performance of un-noised model


Notes:
1. The error `{ENVIRONMENT_PATH}/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`, as required by OpenMM, can be fixed via `conda install -c conda-forge libstdcxx-ng`. See https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found
2. For some reason, getting different accuracies on the multi_pdb test? After every run. I do not understand in the slightest. No way the prediction is stochastic. Maybe strucvtural info is stochastic in some way?


Profiling:
1. Getting structural info takes 80% of the processing time, and (80% * 86%) of the prediction time --> TODO profile get_structural_info

