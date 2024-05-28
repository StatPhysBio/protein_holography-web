

import pyrosetta
init_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 -mute all -include_sugars -ignore_zero_occupancy false -obey_ENDMDL 1'
pyrosetta.init(init_flags, silent=True)

pdb_file = '1BKX.pdb'
pdb_name = pdb_file[:-4]
pose = pyrosetta.pose_from_pdb(pdb_file)
pose.dump_pdb(f'{pdb_name}__pyrosetta_preprocess.pdb')



