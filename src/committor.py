"""
Compute committor probabilities from a discretized MD trajectory.

The committor probability q(i) of cluster i is the probability that a
trajectory starting in cluster i reaches the folded state A before the
unfolded state B. We estimate it by counting, for each visit to cluster i,
whether the next A or B hit comes first ("counting once" strategy:
consecutive revisits to the same cluster are collapsed into one entry).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def compute_committor(cluster_csv, folded_clusters, unfolded_clusters):
    """
    Compute per-cluster committor probabilities from a cluster assignment CSV.

    The CSV must have columns: Time_ps (or similar) and Cluster_ID (in that order).
    Consecutive identical cluster IDs are collapsed (only the first entry of each
    uninterrupted run counts), so the counting reflects genuine transitions.

    Parameters
    ----------
    cluster_csv : str
        Path to the cluster assignment CSV (e.g. cluster_40.csv).
    folded_clusters : list of int
        Cluster IDs that define the folded (A) state.
    unfolded_clusters : list of int
        Cluster IDs that define the unfolded (B) state.

    Returns
    -------
    dict
        {cluster_id: {"A": count_A, "B": count_B, "committor": float}}
        committor = count_A / (count_A + count_B).
        Only clusters that had at least one A or B hit are included.
    """
    data = pd.read_csv(cluster_csv)
    # Accept any column order: first col = time, second = cluster ID
    time_col, cluster_col = data.columns[0], data.columns[1]

    master_count = defaultdict(lambda: {"A": 0, "B": 0})
    count = defaultdict(int)
    prev_cluster = None

    for _, row in data.iterrows():
        cluster = int(row[cluster_col])
        if cluster == prev_cluster:
            continue

        if cluster in folded_clusters:
            for key in count:
                master_count[key]["A"] += count[key]
            count = defaultdict(int)
            prev_cluster = cluster
            continue

        if cluster in unfolded_clusters:
            for key in count:
                master_count[key]["B"] += count[key]
            count = defaultdict(int)
            prev_cluster = cluster
            continue

        count[cluster] += 1
        prev_cluster = cluster

    result = {}
    for cluster_id, hits in master_count.items():
        total = hits["A"] + hits["B"]
        if total == 0:
            continue
        result[cluster_id] = {
            "A": hits["A"],
            "B": hits["B"],
            "committor": hits["A"] / total,
        }
    return result


def write_committor_csv(cluster_csv, output_csv, committor_dict,
                        folded_clusters, unfolded_clusters):
    """
    Append a Committor_prob column to a cluster CSV and write a new file.

    Parameters
    ----------
    cluster_csv : str
        Original cluster CSV path.
    output_csv : str
        Path to write the annotated CSV.
    committor_dict : dict
        Output of compute_committor().
    folded_clusters : list of int
        These receive Committor_prob = 1.0.
    unfolded_clusters : list of int
        These receive Committor_prob = 0.0.
        All other clusters with no transition data receive "Null".
    """
    data = pd.read_csv(cluster_csv)
    cluster_col = data.columns[1]

    def get_prob(cluster_id):
        cid = int(cluster_id)
        if cid in folded_clusters:
            return 1.0
        if cid in unfolded_clusters:
            return 0.0
        if cid in committor_dict:
            return committor_dict[cid]["committor"]
        return "Null"

    data["Committor_prob"] = data[cluster_col].apply(get_prob)
    data.to_csv(output_csv, index=False)
    print(f"Written: {output_csv}")


def plot_committor(committor_dict, title="Committor per cluster"):
    """Scatter plot of committor probability vs. cluster ID."""
    ids = sorted(committor_dict.keys())
    probs = [committor_dict[i]["committor"] for i in ids]

    plt.figure(figsize=(10, 4))
    plt.scatter(ids, probs, s=20)
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Cluster ID")
    plt.ylabel("Committor Probability q(A|i)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Histogram over committor values
    plt.figure(figsize=(6, 3))
    plt.hist(probs, bins=np.arange(0, 1.05, 0.1), color="steelblue", edgecolor="k")
    plt.xlabel("Committor Probability")
    plt.ylabel("Number of clusters")
    plt.title("Distribution of committor values")
    plt.tight_layout()
    plt.show()
