"""
Stratified downsampling by committor probability.

Committor distributions are strongly bimodal (most frames have q≈0 or q≈1).
Uniform random sampling therefore under-represents the transition-state region
(q≈0.5) and leads to poor regression accuracy there. This module bins frames
into equal-width probability intervals and samples the same number from each,
producing a balanced training set.
"""

import numpy as np
import pandas as pd


def dsample(df, down=0.1):
    """
    Stratified downsample a feature+committor DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Last column must be 'Committor_prob' (float, already cleaned of NaN).
    down : float
        Target fraction of the original dataset to retain (0 < down ≤ 1).

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with 'bin' column removed.
    """
    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1])
    labels = [f"{round(bins[i], 1)}-{round(bins[i+1], 1)}" for i in range(len(bins) - 1)]
    n_bins = len(bins) - 1

    df_ = df.copy()
    total_samples = int(len(df_) * down)
    samples_per_bin = total_samples // n_bins

    y_column = df_.columns[-1]
    df_["bin"] = pd.cut(df_[y_column], labels=labels, bins=bins, include_lowest=True)

    sampled_data = []
    for _, bin_group in df_.groupby("bin", observed=True):
        n = min(samples_per_bin, len(bin_group))
        sampled_data.append(bin_group.sample(n=n, random_state=42))

    return pd.concat(sampled_data).reset_index(drop=True).drop(columns=["bin"])
