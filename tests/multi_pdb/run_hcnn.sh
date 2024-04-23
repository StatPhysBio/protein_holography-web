

python -u ../../run_hcnn.py \
                -m HCNN_0p00 \
                -pd pdbs \
                -o multi_pdb_output.csv \
                -r probas logprobas embeddings logits \
                -bs 512 \
                -v 1
                



