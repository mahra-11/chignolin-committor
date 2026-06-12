"""
Extract structural features from an MD trajectory using PyEMMA.

Supported feature types:
  - 'custom_distances'   : pairwise distances for the provided atom-index pairs
  - 'backbone_torsions'  : φ/ψ backbone torsions in cos/sin form
  - 'sidechain_torsions' : sidechain torsions in cos/sin form
  - 'all_distances'      : all Cα-Cα distances (add_distances_ca)
  - 'residue_mindist'    : residue-residue minimum distances
"""

import pandas as pd
import pyemma


def features(pdb, traj_files, pairs=None, al="custom_distances", stride=1):
    """
    Load an MD trajectory and extract a feature matrix.

    Parameters
    ----------
    pdb : str
        Path to the reference PDB/GRO topology file.
    traj_files : list of str
        Path(s) to trajectory file(s) (.xtc format).
    pairs : list of tuple (int, int), optional
        0-based atom index pairs for 'custom_distances'. Required if
        al == 'custom_distances'.
    al : str
        Feature type: 'custom_distances', 'backbone_torsions',
        'sidechain_torsions', 'all_distances', or 'residue_mindist'.
    stride : int
        Load every `stride`-th frame.

    Returns
    -------
    pd.DataFrame
        Feature matrix with PyEMMA feature labels as column names.
    """
    feat = pyemma.coordinates.featurizer(pdb)

    if al == "backbone_torsions":
        feat.add_backbone_torsions(cossin=True)
    elif al == "sidechain_torsions":
        feat.add_sidechain_torsions(cossin=True)
    elif al == "all_distances":
        feat.add_distances_ca()
    elif al == "residue_mindist":
        feat.add_residue_mindist()
    else:  # custom_distances
        if pairs is None:
            raise ValueError("pairs must be provided for 'custom_distances'")
        feat.add_distances(pairs)

    all_data = pyemma.coordinates.load(traj_files, features=feat, stride=stride)
    return pd.DataFrame(all_data, columns=feat.describe())
