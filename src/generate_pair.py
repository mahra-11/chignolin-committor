"""
Generate unique atom index pairs from two lists.
Used to produce the set of (i, j) atom pairs for pairwise distance features.
"""


def pair(A, B):
    """
    Return all unique (i-1, j-1) pairs between elements of A and B,
    skipping self-pairs and duplicates regardless of order.

    Parameters
    ----------
    A, B : list of int
        1-based atom indices (e.g. from ONH()).

    Returns
    -------
    list of tuple (int, int)
        0-based atom index pairs suitable for PyEMMA featurizer.add_distances().
    """
    pairs = []
    for i in A:
        for j in B:
            if (j - 1, i - 1) in pairs:
                continue
            if i == j:
                continue
            pairs.append((i - 1, j - 1))
    return pairs
