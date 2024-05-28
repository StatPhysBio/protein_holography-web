

import os, sys
import io
import tempfile

from Bio.PDB import (
    PDBParser,
    SASA,
)
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from functools import partial
from typing import *
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from Bio.PDB.PDBIO import PDBIO


def remove_waters_pdb(original: str, waterless: str) -> None:
    """
    Removes water atoms from a pdb file

    Original: path to pdb file with original data
    waterless: path where new PDB file will be written
    """
    with io.StringIO() as buffer:
        with open(original, "r") as f_in:
            for line in f_in.readlines():
                if "HOH" not in line: 
                    buffer.write(line)

        with open(waterless, "w") as f_out:
            print(buffer.getvalue(), file=f_out)


pdb_file = '1BKX.pdb'

parser = PDBParser(QUIET=True)

pdb_name = pdb_file[:-4]
L = len(pdb_name)

tmp = tempfile.NamedTemporaryFile()

fixer = PDBFixer(filename=pdb_file)

fixer.findMissingResidues()
fixer.findMissingAtoms()

# fixer.findNonstandardResidues() # not used originally
# fixer.replaceNonstandardResidues() # not used originally

fixer.addMissingAtoms()

fixer.addMissingHydrogens()

# fixer.removeHeterogens(keepWater=False) # not used originally

with open(tmp.name, "w") as w:
    PDBFile.writeFile(fixer.topology, fixer.positions, file=w, keepIds=True)

remove_waters_pdb(original=tmp.name, waterless=tmp.name) # only necessary if hydrogens are added?

pdb_file = tmp.name

structure = parser.get_structure(pdb_name, pdb_file)

pdbio = PDBIO()
pdbio.set_structure(structure)
pdbio.save(f'{pdb_name}__pdbfixer_preprocess.pdb')


