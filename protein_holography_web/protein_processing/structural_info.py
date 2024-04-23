
import os, sys
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

from protein_holography_web.protein_processing.constants import aa_to_one_letter, CHARGES_AMBER99SB, VEC_AA_ATOM_DICT

import numpy as np
import numpy.typing as npt

from protein_holography_web.utils import log_config as logging

logger = logging.getLogger(__name__)


def get_structural_info(pdb_file: str,
                        padded_length: Optional[int] = None,
                        SASA: bool = True,
                        charge: bool = True,
                        DSSP: bool = True,
                        angles: bool = True,
                        fix: bool = False,
                        hydrogens: bool = False,
                        multi_struct: str = "warn"):

    """
    Get structural info from a single pdb file.
    If padded_length is None, does not pad the protein.
    """

    if isinstance(pdb_file, str):
        L = len(pdb_file.split('/')[-1].split('.')[0])
    else:
        L = len(pdb_file[0].split('/')[-1].split('.')[0])
        for i in range(1, len(pdb_file)):
            L = max(L, len(pdb_file[i].split('/')[-1].split('.')[0]))

    if isinstance(pdb_file, str):
        pdb_file = [pdb_file]
    

    n = 0
    for i, pdb_file in enumerate(pdb_file):

        if padded_length is None:
            si = get_structural_info_from_protein(
                    pdb_file,
                    calculate_SASA=SASA,
                    calculate_charge=charge,
                    calculate_DSSP=DSSP,
                    calculate_angles=angles,
                    fix=fix,
                    hydrogens=hydrogens,
                    multi_struct=multi_struct)
        else:
            si = get_padded_structural_info(
                    pdb_file,
                    padded_length=padded_length,
                    SASA=SASA,
                    charge=charge,
                    DSSP=DSSP,
                    angles=angles,
                    fix=fix,
                    hydrogens=hydrogens,
                    multi_struct=multi_struct)

        if si[0] is None:
            print(f"Failed to process {pdb_file}", file=sys.stderr)
            continue

        try:
            pdb,atom_names,elements,res_ids,coords,sasas,charges,res_ids_per_residue,angles,vecs = si
        except ValueError:
            pdb,(atom_names,elements,res_ids,coords,sasas,charges,res_ids_per_residue,angles,vecs) = si

        if n == 0:
            if padded_length is None:
                length = len(atom_names)
                dt = np.dtype([
                    ('pdb',f'S{L}',()),
                    ('atom_names', 'S4', (length)),
                    ('elements', 'S2', (length)),
                    ('res_ids', f'S{L}', (length, 6)),
                    ('coords', 'f4', (length, 3)),
                    ('SASAs', 'f4', (length)),
                    ('charges', 'f4', (length)),
                ])
            else:
                dt = np.dtype([
                    ('pdb',f'S{L}',()),
                    ('atom_names', 'S4', (padded_length)),
                    ('elements', 'S2', (padded_length)),
                    ('res_ids', f'S{L}', (padded_length, 6)),
                    ('coords', 'f4', (padded_length, 3)),
                    ('SASAs', 'f4', (padded_length)),
                    ('charges', 'f4', (padded_length)),
                ])
            
            np_protein = np.zeros(shape=(len(pdb_file),), dtype=dt)

        np_protein[n] = (pdb,atom_names,elements,res_ids,coords,sasas,charges,)
        
        n += 1

    np_protein.resize((n,))

    return np_protein



def get_padded_structural_info(
    pdb_file: str,
    padded_length: int = 200000,
    SASA: bool = True,
    charge: bool = True,
    DSSP: bool = True,
    angles: bool = True,
    fix: bool = False,
    hydrogens: bool = False,
    multi_struct: str = "warn",
) -> Tuple[
    bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Extract structural info used for holographic projection from Biopython pose.

    Parameters
    ----------
    pdb_file: path to file with pdb
    padded_length: size to pad to
    SASA: Whether or not to calculate SASA
    charge: Whether or not to calculate charge
    DSSP: Whether or not to calculate DSSP
    angles: Whether or not to calculate anglges
    Fix: Whether or not to fix missing atoms
    Hydrogens: Whether or not to add hydrogen atoms
    multi_struct: Behavior for handling PDBs with multiple structures

    Returns
    -------
    tuple of (bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
              np.ndarray)
        The entries in the tuple are
            bytes encoding the pdb name string
            bytes array encoding the atom names of shape [max_atoms]
            bytes array encoding the elements of shape [max_atoms]
            bytes array encoding the residue ids of shape [max_atoms,6]
            float array of shape [max_atoms,3] representing the 3D Cartesian
              coordinates of each atom
            float array of shape [max_atoms] storing the SASA of each atom
            float array of shape [max_atoms] storing the partial charge of each atom
    """

    try:
        pdb, ragged_structural_info = get_structural_info_from_protein(
            pdb_file,
            calculate_SASA=SASA,
            calculate_charge=charge,
            calculate_DSSP=DSSP,
            calculate_angles=angles,
            fix=fix,
            hydrogens=hydrogens,
            multi_struct=multi_struct,
        )

        mat_structural_info = pad_structural_info(
            ragged_structural_info, padded_length=padded_length
        )
    except Exception as e:
        logger.error(f"Failed to process {pdb_file}")
        logger.error(e)
        return (None,)

    return (pdb, *mat_structural_info)


def get_structural_info_from_protein(
    pdb_file: str,
    remove_nonwater_hetero: bool = False,
    remove_waters: bool = True,
    calculate_SASA: bool = True,
    calculate_charge: bool = True,
    calculate_DSSP: bool = True,
    calculate_angles: bool = True,
    fix: bool = False,
    hydrogens: bool = False,
    multi_struct: str = "warn",
) -> Tuple[str, Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    """
    Params:
        - pdb_file: path to pdb file
        - remove_nonwater_hetero, remove_waters: whether or not to remove certain atoms
        - calculate_X: if set to false, go faster
        - if set, will fix missing atoms in pdb
        - hydrogens: if set, will add hydrogens to pdb
        - multi_struct: Behavior for handling PDBs with multiple structures

    Returns:
        Tuple of (pdb, (atom_names, elements, res_ids, coords, sasas, charges, res_ids_per_residue, angles, norm_vecs, is_multi_model [1 or 0] ))

    By default, biopyton selects only atoms with the highest occupancy, thus behaving like pyrosetta does with the flag "-ignore_zero_occupancy false"
    """
    parser = PDBParser(QUIET=True)

    pdb_name = pdb_file[:-4]
    L = len(pdb_name)

    if fix or hydrogens:
        tmp = tempfile.NamedTemporaryFile()

        fixer = PDBFixer(filename=pdb_file)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        if hydrogens:
            fixer.addMissingHydrogens()
        
        fixer.removeHeterogens(keepWater=False)

        with open(tmp.name, "w") as w:
            PDBFile.writeFile(fixer.topology, fixer.positions, file=w, keepIds=True)

        pdb_file = tmp.name

    structure = parser.get_structure(pdb_name, pdb_file)

    # assume the pdb name was provided as id to create the structure
    pdb = structure.get_id()
    pdb = os.path.basename(pdb).replace("_", "-") # "_" is reserved for the res_id delimiter later in pipeline 

    models = list(structure.get_models())
    if len(models) != 1:
        if multi_struct == "crash":
            raise ValueError(f"More than 1 model found for {pdb_file}")
        else:
            if multi_struct == "warn":
                logger.warn(
                    f"{len(models)} models found for {pdb_file}. Setting structure to the first model."
                )
            structure = models[0]

    if calculate_SASA:
        # Calculates SASAs with biopython; each atom will have a .sasa attribute
        SASA.ShrakeRupley().compute(structure, level="A")

    if calculate_DSSP:
        dssp_dict, _dssp_keys = dssp_dict_from_pdb_file(pdb_file)
    else:
        dssp_dict = {}

    if fix or hydrogens:
        tmp.close()

    # lists for each type of information to obtain
    atom_names = []
    elements = []
    sasas = []
    coords = []
    charges = []
    res_ids = []

    angles = []
    vecs = []
    res_ids_per_residue = []

    chi_atoms = {}

    def pad_for_consistency(string):
        return string.ljust(4, " ")

    # get structural info from each residue in the protein
    for atom in structure.get_atoms():
        atom_full_id = atom.get_full_id()

        if remove_waters and atom_full_id[3][0] == "W":
            continue

        if remove_nonwater_hetero and atom_full_id[3][0] not in {" " "W"}:
            continue

        chain = atom_full_id[2]
        resnum = atom_full_id[3][1]
        icode = atom_full_id[3][2]
        atom_name_unpadded = atom.get_name()
        atom_name = pad_for_consistency(atom_name_unpadded)
        element = atom.element
        coord = atom.get_coord()
        ss = dssp_dict.get((chain, (" ", resnum, " ")), ("_", "null"))[1]

        residue = atom.get_parent().resname
        if residue in aa_to_one_letter:
            aa = aa_to_one_letter[residue]
        else:
            aa = "Z"

        res_id = np.array([aa, pdb, chain, resnum, icode, ss], dtype=f"S{L}")

        res_key = tuple(res_id)
        if res_key not in chi_atoms:
            chi_atoms[res_key] = residue, {}
        chi_atoms[res_key][1][atom_name_unpadded] = coord

        atom_names.append(atom_name)
        elements.append(element)
        res_ids.append(res_id)
        coords.append(coord)
        if calculate_SASA:
            sasas.append(atom.sasa)

        if calculate_charge:
            res_charges = CHARGES_AMBER99SB[residue]
            if isinstance(res_charges, dict):
                charge = res_charges[atom_name_unpadded.upper()]
            elif isinstance(res_charges, float) or isinstance(res_charges, int):
                charge = res_charges
            else:
                raise ValueError(
                    f"Unknown charge type: {type(res_charges)}. Something must be wrong."
                )
            charges.append(charge)

    if calculate_angles:
        for res_key, (resname, atoms) in chi_atoms.items():
            res_ids_per_residue.append(np.array([*res_key], dtype=f"S{L}"))
            chis, norms = get_chi_angles_and_norm_vecs(resname, atoms, pdb)
            angles.append(chis)
            vecs.append(norms)

    atom_names = np.array(atom_names, dtype="|S4")
    elements = np.array(elements, dtype="S1")
    coords = np.array(coords)
    res_ids = np.array(res_ids)
    sasas = np.array(sasas)
    charges = np.array(charges)
    res_ids_per_residue = np.array(res_ids_per_residue)
    angles = np.array(angles)
    vecs = np.array(vecs)

    return pdb, (
        atom_names,
        elements,
        res_ids,
        coords,
        sasas,
        charges,
        res_ids_per_residue,
        angles,
        vecs,
    )


def get_chi_angle(
    plane_norm_1: npt.NDArray,
    plane_norm_2: npt.NDArray,
    a2: npt.NDArray,
    a3: npt.NDArray,
) -> float:
    """
    Calculates the dihedral angle given two plane norms and a2, a3 atom places
    """
    eps = 1e-6

    sign_vec = a3 - a2
    sign_with_magnitude = np.dot(sign_vec, np.cross(plane_norm_1, plane_norm_2))
    sign = sign_with_magnitude / (np.abs(sign_with_magnitude) + eps)

    dot = np.dot(plane_norm_1, plane_norm_2) / (
        np.linalg.norm(plane_norm_1) * np.linalg.norm(plane_norm_2)
    )
    chi_angle = sign * np.arccos(dot * (1 - eps))

    return np.degrees(chi_angle)


def get_chi_angles_and_norm_vecs(
    resname: str, residue: Dict[str, npt.NDArray], pdb: str
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Get chi angles and normal vectors (which are used to compute chi angles) from a residue.
    Uses the tables available at http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html

    Parameters
    ----------
    resname : str
        name of the residue
    residue : Dict[str, npt.NDArray]
        Dictionary mapping atom name to coords
    pdb: str
        Name of protein (logging use only)

    Returns
    -------
    np.ndarray
        The chi angles (4 of them)
        Will be nan if there are no vectors for the residue there

    np.ndarray
        The normal vectors (5 of them as the CB one is included)
        Will be nan if there are no vectors for the residue there
    """
    vecs = np.full((5, 3), np.nan, dtype=float)
    chis = np.full(4, np.nan, dtype=float)
    atom_names = VEC_AA_ATOM_DICT.get(resname)
    try:
        if atom_names is not None:
            for i in range(len(atom_names)):
                p1 = residue[atom_names[i][0]]
                p2 = residue[atom_names[i][1]]
                p3 = residue[atom_names[i][2]]
                v1 = p1 - p2
                v2 = p3 - p2
                # v1 = p1 - p2
                # v2 = p1 - p3
                x = np.cross(v1, v2)
                vecs[i] = x / np.linalg.norm(x)

            for i in range(len(atom_names) - 1):
                chis[i] = get_chi_angle(
                    vecs[i],
                    vecs[i + 1],
                    residue[atom_names[i][1]],
                    residue[atom_names[i][2]],
                )
    except Exception:
        logger.warning(
            (
                f"Failed to calculate chi angles/normal vectors for a {resname} in {pdb} with the error below. "
                "The remaining structural info for this protein, including other chi angles, is likely still valid."
            ),
            exc_info=True,
        )
        # returns chis and vecs empty
        vecs = np.full((5, 3), np.nan, dtype=float)
        chis = np.full(4, np.nan, dtype=float)

    return chis, vecs


def pad(arr: npt.NDArray, padded_length: int = 100) -> npt.NDArray:
    """
    Pad an array long axis 0

    Parameters
    ----------
    arr : npt.NDArray
    padded_length : int

    Returns
    -------
    npt.NDArray
    """
    # get dtype of input array
    dt = arr.dtype

    # shape of sub arrays and first dimension (to be padded)
    shape = arr.shape[1:]
    orig_length = arr.shape[0]

    # check that the padding is large enough to accomdate the data
    if padded_length < orig_length:
        logger.warn(
            f"Error: Padded length of {padded_length}",
            f"is smaller than original length of array {orig_length}",
        )

    # create padded array
    padded_shape = (padded_length, *shape)
    mat_arr = np.zeros(padded_shape, dtype=dt)

    # add data to padded array
    mat_arr[:orig_length] = np.array(arr)

    return mat_arr


def pad_structural_info(
    ragged_structure: Tuple[npt.NDArray, ...], padded_length: int = 100
) -> List[npt.NDArray]:
    """
    Pad structural into arrays
    """
    pad_custom = partial(pad, padded_length=padded_length)
    mat_structure = list(map(pad_custom, ragged_structure))

    return mat_structure