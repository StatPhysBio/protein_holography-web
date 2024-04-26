

python -u ../../run_hcnn_on_pdbfiles.py \
                -m HCNN_0p00 \
                -pd pdbs \
                -o single_pdb_output.csv \
                -r probas logprobas embeddings logits \
                -bs 512 \
                -v 1
                



