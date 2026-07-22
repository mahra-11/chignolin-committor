"""
Order-parameter (Q, Rg) committor estimation — the "Changing to Q" method.

The original pipeline (src/committor.py) discretizes the trajectory into
TICA/K-Means clusters and estimates the committor per cluster. As shown in
the presentation, that approach has two problems: (1) clustering on raw
coordinates can put geometrically distinct conformers in the same cluster,
and (2) the folded state dominates the cluster population so heavily that
the boundary between "folded" and "unfolded" had to be RMSD-filtered,
throwing away >60% of the frames.

This module replaces clustering with two continuous, physically meaningful
order parameters:

  - Q  : fraction of native contacts (Best et al. 2013 formula), 0 = fully
         unfolded, 1 = fully folded.
  - Rg : radius of gyration.

No clustering is performed. Instead:

  1. Every frame gets a (Q, Rg) pair.
  2. Frames are binned onto a 2D (Rg, Q) grid ("group similar moments").
  3. Each frame is labeled by a visit-based first-passage rule: walking
     forward along its own trajectory, does it hit the folded boundary
     (Q >= q_folded) or the unfolded boundary (Q <= q_unfolded) first?
     ("count the outcomes")
  4. The per-box committor is the mean of the per-frame outcomes of the
     frames that fall in that box, and can be used as a regression target
     for an ML model that predicts committor from structural features
     ("train a model to guess the odds").

This reproduces Figure 6 (Q vs. Rg heat map) and Figure 7 (histogram of
committor values, weighted by frame count) from the presentation.
"""

import numpy as np
import pandas as pd


def compute_native_contacts(traj, reference, cutoff=0.45, beta=50.0, lam=1.8,
                             min_seq_separation=3):
    """
    Fraction of native contacts Q(X) using the smooth Best et al. (2013)
    switching function:

        Q(X) = (1/N) * sum_{(i,j) in native} 1 / (1 + exp(beta * (r_ij(X) - lam * r_ij_0)))

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory to score (all frames).
    reference : mdtraj.Trajectory
        Single-frame reference structure defining the native (folded) contacts.
    cutoff : float
        Distance (nm) below which a residue pair in `reference` counts as a
        native contact.
    beta : float
        Smoothing steepness (nm^-1); higher = closer to a hard cutoff.
    lam : float
        Tolerance multiplier on the native distance.
    min_seq_separation : int
        Minimum residue index separation for a pair to be considered
        (excludes trivially-close backbone neighbors).

    Returns
    -------
    np.ndarray, shape (n_frames,)
        Q for every frame of `traj`.
    """
    import mdtraj as md

    ca = reference.topology.select("name CA")
    pairs = np.array(
        [(i, j) for idx, i in enumerate(ca) for j in ca[idx + 1:]
         if abs(reference.topology.atom(i).residue.index
                - reference.topology.atom(j).residue.index) >= min_seq_separation]
    )

    ref_dist = md.compute_distances(reference, pairs)[0]
    native = pairs[ref_dist < cutoff]
    native_r0 = ref_dist[ref_dist < cutoff]

    if len(native) == 0:
        raise ValueError("No native contacts found — check `cutoff` and topology.")

    dist = md.compute_distances(traj, native)
    q = 1.0 / (1.0 + np.exp(beta * (dist - lam * native_r0)))
    return q.mean(axis=1)


def compute_rg(traj):
    """Radius of gyration (nm) for every frame."""
    import mdtraj as md
    return md.compute_rg(traj)


def assign_boundary_states(q, q_folded=0.9, q_unfolded=0.1):
    """
    Label each frame as folded (+1), unfolded (-1), or transition (0)
    from its Q value alone.

    Parameters
    ----------
    q : np.ndarray
        Per-frame fraction of native contacts.
    q_folded, q_unfolded : float
        Thresholds defining the folded/unfolded boundary states
        (defaults match the presentation: folded=0.9, unfolded=0.1).

    Returns
    -------
    np.ndarray of int8, same shape as `q`.
    """
    state = np.zeros(len(q), dtype=np.int8)
    state[q >= q_folded] = 1
    state[q <= q_unfolded] = -1
    return state


def visit_based_committor(state, trajectory_id=None):
    """
    Per-frame first-passage committor label.

    For every transition frame (state == 0), walk forward in time within
    the same trajectory until the next boundary frame (state == +-1) is
    reached. The label is 1 if that boundary is folded, 0 if unfolded.
    Boundary frames are labeled with their own state (1 -> 1.0, -1 -> 0.0).
    Frames that never reach a boundary before the trajectory ends (or a
    trajectory-id change) are labeled NaN and excluded downstream.

    This is the "visit-based" rule referenced in Figure 7: it is a single
    stochastic 0/1 realization per frame, not yet a probability — it only
    becomes a committor estimate once many frames are averaged together
    (e.g. within one (Rg, Q) box, see `bin_committor_2d`).

    Parameters
    ----------
    state : np.ndarray of int8
        Output of `assign_boundary_states`, in trajectory time order.
    trajectory_id : np.ndarray, optional
        Same length as `state`; frames only look forward within a
        contiguous run of the same id. If None, the whole array is
        treated as one trajectory.

    Returns
    -------
    np.ndarray of float64, shape (len(state),)
        1.0 = folded-first, 0.0 = unfolded-first, NaN = never resolved.
    """
    n = len(state)
    if trajectory_id is None:
        trajectory_id = np.zeros(n, dtype=np.int64)

    label = np.full(n, np.nan, dtype=np.float64)

    # Walk backward once: carry forward the most recent boundary hit,
    # resetting whenever the trajectory id changes.
    next_boundary = np.nan
    current_traj = None
    for i in range(n - 1, -1, -1):
        if trajectory_id[i] != current_traj:
            next_boundary = np.nan
            current_traj = trajectory_id[i]

        if state[i] == 1:
            next_boundary = 1.0
        elif state[i] == -1:
            next_boundary = 0.0

        label[i] = next_boundary

    return label


def bin_committor_2d(rg, q, committor_label, n_bins=60,
                      rg_range=None, q_range=None):
    """
    Bin frames onto a 2D (Rg, Q) grid and compute the box-averaged
    committor, reproducing Figures 6 and 7.

    Parameters
    ----------
    rg, q : np.ndarray
        Per-frame radius of gyration and fraction of native contacts.
    committor_label : np.ndarray
        Per-frame visit-based label from `visit_based_committor`
        (NaNs are dropped before binning).
    n_bins : int or (int, int)
        Number of bins along (Rg, Q).
    rg_range, q_range : (float, float), optional
        Axis limits; defaults to the data range.

    Returns
    -------
    dict with:
        'frame_counts' : 2D np.ndarray (n_bins_rg, n_bins_q) — Figure 6 data
        'box_committor': 2D np.ndarray, NaN where a box is empty
        'rg_edges', 'q_edges' : 1D np.ndarray bin edges
        'box_committor_frames' : pd.DataFrame with one row per *occupied*
            box: rg_center, q_center, n_frames, committor — the data
            underlying Figure 7 (histogram of `committor`, weighted by
            `n_frames`).
    """
    mask = ~np.isnan(committor_label)
    rg_f, q_f, lab_f = rg[mask], q[mask], committor_label[mask]

    if isinstance(n_bins, int):
        n_bins = (n_bins, n_bins)

    rg_range = rg_range or (rg.min(), rg.max())
    q_range = q_range or (q.min(), q.max())

    rg_edges = np.linspace(*rg_range, n_bins[0] + 1)
    q_edges = np.linspace(*q_range, n_bins[1] + 1)

    frame_counts, _, _ = np.histogram2d(rg_f, q_f, bins=[rg_edges, q_edges])
    committor_sum, _, _ = np.histogram2d(rg_f, q_f, bins=[rg_edges, q_edges],
                                          weights=lab_f)

    with np.errstate(invalid="ignore", divide="ignore"):
        box_committor = np.where(frame_counts > 0, committor_sum / frame_counts, np.nan)

    rg_idx, q_idx = np.nonzero(frame_counts)
    rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])

    box_df = pd.DataFrame({
        "rg_center": rg_centers[rg_idx],
        "q_center": q_centers[q_idx],
        "n_frames": frame_counts[rg_idx, q_idx].astype(int),
        "committor": box_committor[rg_idx, q_idx],
    })

    return {
        "frame_counts": frame_counts,
        "box_committor": box_committor,
        "rg_edges": rg_edges,
        "q_edges": q_edges,
        "box_committor_frames": box_df,
    }


def plot_qrg_heatmap(binned, ax=None, cmap="viridis"):
    """Figure 6: log-scale heat map of frame counts over (Rg, Q)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    counts = binned["frame_counts"].T  # Q on y-axis
    counts = np.ma.masked_where(counts == 0, counts)
    mesh = ax.pcolormesh(binned["rg_edges"], binned["q_edges"], counts,
                          norm=LogNorm(), cmap=cmap, shading="flat")
    plt.colorbar(mesh, ax=ax, label="Frame count (log scale)")
    ax.set_xlabel("Radius of gyration (nm)")
    ax.set_ylabel("Q (fraction of native contacts)")
    ax.set_title("Q vs. Rg, all frames, log color scale")
    return ax


def plot_committor_histogram(binned, bins=20, ax=None):
    """Figure 7: histogram of per-box committor values, weighted by frame count."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    df = binned["box_committor_frames"]
    ax.hist(df["committor"], bins=bins, weights=df["n_frames"],
            color="steelblue", edgecolor="k")
    ax.set_yscale("log")
    ax.set_xlabel("Committor value (P(folded first))")
    ax.set_ylabel("Number of frames (log scale)")
    ax.set_title("Frame count by committor value (visit-based)")
    return ax
