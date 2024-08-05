
import os
import argparse

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

HERMES_NAME_TO_OLD_HCNN_NAME = {
    "HERMES_Bp_000": "HCNN_biopython_proteinnet_extra_mols_0p00",
    "HERMES_Bp_050": "HCNN_biopython_proteinnet_extra_mols_0p50",
    "HERMES_Py_000": "HCNN_pyrosetta_proteinnet_extra_mols_0p00",
    "HERMES_Py_050": "HCNN_pyrosetta_proteinnet_extra_mols_0p50",
    "HERMES_Bp_000_FT_Ros_ddG": "HCNN_biopython_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all",
    "HERMES_Bp_050_FT_Ros_ddG": "HCNN_biopython_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all",
    "HERMES_Py_000_FT_Ros_ddG": "HCNN_pyrosetta_proteinnet_extra_mols_0p00_finetuned_with_rosetta_ddg_all",
    "HERMES_Py_050_FT_Ros_ddG": "HCNN_pyrosetta_proteinnet_extra_mols_0p50_finetuned_with_rosetta_ddg_with_0p50_all"
}

SLURM_SETUP = "#!/bin/bash\n\
#SBATCH --job-name={jobname}\n\
#SBATCH --account={account}\n\
#SBATCH --partition={partition}\n{gpu_text}\
#SBATCH --nodes=1\n\
#SBATCH --ntasks-per-node={num_cores}\n\
#SBATCH --time={walltime}\n\
#SBATCH --mem={memory}\n{email_text}\
#SBATCH -e {dumpfiles_folder}/{jobname}.err\n\
#SBATCH -o {dumpfiles_folder}/{jobname}.out"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-pd', '--folder_with_pdbs', type=str, default=None,
                        help='Directory containing PDB files to run inference on. Inference is run on all sites in the structure.')
    
    parser.add_argument('-df', '--dumpfiles_folder', type=str, default='./dumpfiles/',
                        help='Root to store dumpfiles.')
    
    parser.add_argument('-of', '--output_folder', type=str, default='./raw_output/',
                        help='Root to store outputs.')
    
    parser.add_argument('-hf', '--hermes_folder', type=str, default=THIS_FOLDER,
                        help='Path to the Hermes folder, containing the run_hcnn_on_pdbfiles.py script.')
    
    parser.add_argument('-A',  '--account', type=str, default='stf')
    parser.add_argument('-P',  '--partition', type=str, default='compute')
    parser.add_argument('-G',  '--use_gpu', type=int, default=0, choices=[0, 1])
    parser.add_argument('-C',  '--num_cores', type=int, default=1)
    parser.add_argument('-W',  '--walltime', type=str, default='5:00:00')
    parser.add_argument('-M',  '--memory', type=str, default='16G')
    parser.add_argument('-E',  '--send_emails', type=int, default=0, choices=[0, 1])
    parser.add_argument('-EA', '--email_address', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.dumpfiles_folder, exist_ok=True)


    if args.use_gpu:
        gpu_text = '#SBATCH --gres=gpu:1\n'
    else:
        gpu_text = ''
    
    if args.send_emails:
        email_text = f'#SBATCH --mail-type=ALL\n#SBATCH --mail-user={args.email_address}\n#SBATCH --export=all\n'
    else:
        email_text = ''
    
    
    pdb_ids = [pdb_file.split('.')[0] for pdb_file in os.listdir(args.folder_with_pdbs)]
    
    for hermes_name, old_hcnn_name in HERMES_NAME_TO_OLD_HCNN_NAME.items():

        output_folder = os.path.join(args.output_folder, hermes_name)
        os.makedirs(output_folder, exist_ok=True)


        for pdb_id in pdb_ids:
            jobname = f'{hermes_name}__{pdb_id}'

            file_with_pdbids_and_chains = f'{args.dumpfiles_folder}/{jobname}__pdbs_and_chains.txt'
            with open(file_with_pdbids_and_chains, 'w') as f:
                f.write(f'{pdb_id}\n')
            
            output_filepath = f'{output_folder}/{hermes_name}__{pdb_id}.csv'

            slurm_script = SLURM_SETUP.format(
                jobname=jobname,
                account=args.account,
                partition=args.partition,
                gpu_text=gpu_text,
                num_cores=args.num_cores,
                walltime=args.walltime,
                memory=args.memory,
                email_text=email_text,
                dumpfiles_folder=args.dumpfiles_folder
            )

            slurm_script += f'\npython {args.hermes_folder}/run_hcnn_on_pdbfiles.py \
                                    -m {old_hcnn_name} \
                                    -pd {args.folder_with_pdbs}\
                                    -pn {file_with_pdbids_and_chains} \
                                    -o {output_filepath} \
                                    -r probas logprobas logits \
                                    -v 1 \
                                    -bs 512'
            
            slurm_script_filepath = 'job.slurm'
            with open(slurm_script_filepath, 'w') as f:
                f.write(slurm_script)
            os.system(f'sbatch {slurm_script_filepath}')

    


            

    





