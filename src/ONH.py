"""
Identify O/N heavy atoms and polar hydrogens in a PDB structure.
These atom indices are used to generate physically meaningful
pairwise distance features (hydrogen-bond-relevant pairs).
"""

from Bio import PDB


def ONH(file):
    """
    Parse a PDB file and return indices of O/N heavy atoms and polar hydrogens.

    Parameters
    ----------
    file : str
        Path to the PDB file.

    Returns
    -------
    oxygen_nitrogen_atoms_index : list of int
        1-based indices of all O and N heavy atoms.
    polar_hydrogen_atoms_index : list of int
        1-based indices of H atoms bonded to O or N (within 1.2 Å).
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("molecule", file)

    oxygen_nitrogen_atoms_index = []
    polar_hydrogen_atoms_index = []

    for model in structure:
        for chain in model:
            i = 0
            for residue in chain:
                for atom in residue:
                    i += 1
                    element = atom.element
                    if element in ["O", "N"]:
                        oxygen_nitrogen_atoms_index.append(i)
                    if element == "H":
                        j = 0
                        for neighbor_residue in chain:
                            for neighbor in neighbor_residue:
                                j += 1
                                if neighbor.element in ["O", "N"]:
                                    bond_distance = atom - neighbor
                                    if bond_distance < 1.2:
                                        if j not in polar_hydrogen_atoms_index:
                                            polar_hydrogen_atoms_index.append(j)
                                        break

    return oxygen_nitrogen_atoms_index, polar_hydrogen_atoms_index
