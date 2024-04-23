
import os
import numpy as np
import re
from collections import defaultdict

N  = b'N   '
CA = b'CA  '
C  = b'C   '
O  = b'O   '
CB = b'CB  '
EMPTY_ATOM_NAME = b''

BACKBONE_ATOMS = np.array([N, CA, C, O])

CHI_ANGLES = {
    'ARG' : [['N','CA','CB','CG'], ['CA','CB','CG','CD'], ['CB','CG','CD','NE'], ['CG','CD','NE','CZ']], #, ['CD','NE','CZ','NH1']],
    'ASN' : [['N','CA','CB','CG'], ['CA','CB','CG','OD1']],
    'ASP' : [['N','CA','CB','CG'], ['CA','CB','CG','OD1']],
    'CYS' : [['N','CA','CB','SG']],
    'GLN' : [['N','CA','CB','CG'], ['CA','CB','CG','CD'], ['CB','CG','CD','OE1']],
    'GLU' : [['N','CA','CB','CG'], ['CA','CB','CG','CD'], ['CB','CG','CD','OE1']],
    'HIS' : [['N','CA','CB','CG'], ['CA', 'CB','CG','ND1']],
    'ILE' : [['N','CA','CB','CG1'], ['CA','CB','CG1','CD1']],
    'LEU' : [['N','CA','CB','CG'], ['CA','CB','CG','CD1']],
    'LYS' : [['N','CA','CB','CG'], ['CA','CB','CG','CD'], ['CB','CG','CD','CE'], ['CG','CD','CE','NZ']],
    'MET' : [['N','CA','CB','CG'], ['CA','CB','CG','SD'], ['CB','CG','SD','CE']],
    'PHE' : [['N','CA','CB','CG'], ['CA','CB','CG','CD1']],
    'PRO' : [['N','CA','CB','CG'], ['CA','CB','CG','CD']],
    'SER' : [['N','CA','CB','OG']],
    'THR' : [['N','CA','CB','OG1']],
    'TRP' : [['N','CA','CB','CG'], ['CA','CB','CG','CD1']],
    'TYR' : [['N','CA','CB','CG'], ['CA','CB','CG','CD1']],
    'VAL' : [['N','CA','CB','CG1']]
}

# transform chi angle atoms by left-justifying anf converting to bytes
from protein_holography_web.utils.protein_naming import aa_to_one_letter
CHI_ATOMS = {aa_to_one_letter[aa]: [CB] + [atoms[-1].ljust(4).encode('utf-8') for atoms in CHI_ANGLES[aa]] for aa in CHI_ANGLES}


##################### Copied from https://github.com/nekitmm/DLPacker/blob/main/utils.py
# read in the charges from special file
CHARGES_AMBER99SB = defaultdict(lambda: 0)  # output 0 if the key is absent
with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "charges.rtp"), "r"
) as f:
    for line in f:
        if line[0] == "[" or line[0] == " ":
            if re.match("\A\[ .{1,3} \]\Z", line[:-1]):
                key = re.match("\A\[ (.{1,3}) \]\Z", line[:-1])[1]
                CHARGES_AMBER99SB[key] = defaultdict(lambda: 0)
            else:
                l = re.split(r" +", line[:-1])
                CHARGES_AMBER99SB[key][l[1]] = float(l[3])
################################################################################


aa_to_one_letter = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

# fmt: off
VEC_AA_ATOM_DICT = {
    'ARG' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','NE'], ['CD','NE','CZ']], #, ['NE','CZ','NH1']],
    'ASN' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','OD1']],
    'ASP' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','OD1']],
    'CYS' : [['N','CA','CB'], ['CA','CB','SG']],
    'GLN' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','OE1']],
    'GLU' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','OE1']],
    'HIS' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','ND1']],
    'ILE' : [['N','CA','CB'], ['CA','CB','CG1'], ['CB','CG1','CD1']],
    'LEU' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'LYS' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD'], ['CG','CD','CE'], ['CD','CE','NZ']],
    'MET' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','SD'], ['CG','SD','CE']],
    'PHE' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'PRO' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD']],
    'SER' : [['N','CA','CB'], ['CA','CB','OG']],
    'THR' : [['N','CA','CB'], ['CA','CB','OG1']],
    'TRP' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'TYR' : [['N','CA','CB'], ['CA','CB','CG'], ['CB','CG','CD1']],
    'VAL' : [['N','CA','CB'], ['CA','CB','CG1']]
}
# fmt: on
