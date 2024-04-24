# protein_holography-web

TODO:
1. Add `requirements.txt` and general tips on installation procedure
2. Profile and speed up inference. Currently takes 20/30 seconds per protein, which is a little too long.
3. Add quick running of T4-Lysozymes and SKEMPI (at least a subset) ddG prediction to check zero-shot performance


Notes:
1. The error `{ENVIRONMENT_PATH}/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`, as required by OpenMM, can be fixed via `conda install -c conda-forge libstdcxx-ng`. See https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found




