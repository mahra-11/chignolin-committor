#!/usr/bin/env python3
"""
End-to-end Q/Rg committor pipeline (the "Changing to Q" method).

Replaces the clustering-based committor estimate (src/committor.py,
notebooks 01-02) with the continuous order-parameter approach from
src/qrg_committor.py: no clustering, a visit-based first-passage rule,
and a (Rg, Q) box-averaged committor — reproducing Figures 6 and 7 of the
presentation. It then trains a small regression model to predict the
committor from (Rg, Q), the "train a model to guess the odds" step.

Usage
-----
    python scripts/run_qrg_committor.py --config config/chignolin.yaml

Requires mdtraj to load the trajectory; see requirements.txt.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrg_committor import (
    assign_boundary_states,
    bin_committor_2d,
    compute_native_contacts,
    compute_rg,
    plot_committor_histogram,
    plot_qrg_heatmap,
    visit_based_committor,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="e.g. config/chignolin.yaml")
    p.add_argument("--q-folded", type=float, default=0.9,
                    help="Q threshold for the folded boundary state")
    p.add_argument("--q-unfolded", type=float, default=0.1,
                    help="Q threshold for the unfolded boundary state")
    p.add_argument("--n-bins", type=int, default=60,
                    help="Number of bins per axis of the (Rg, Q) grid")
    p.add_argument("--stride", type=int, default=None,
                    help="Override the config's stride (frame subsampling)")
    p.add_argument("--out-dir", default="results", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    import mdtraj as md

    pdb = cfg["topology_pdb"]
    traj_files = cfg["trajectory_xtc"]
    stride = args.stride if args.stride is not None else cfg.get("stride", 1)

    print(f"Loading trajectory (stride={stride}) …")
    traj = md.load(traj_files, top=pdb, stride=stride)
    reference = md.load(pdb)

    print("Computing Q (fraction of native contacts) and Rg …")
    q = compute_native_contacts(traj, reference)
    rg = compute_rg(traj)

    state = assign_boundary_states(q, q_folded=args.q_folded, q_unfolded=args.q_unfolded)
    n_folded = np.sum(state == 1)
    n_unfolded = np.sum(state == -1)
    print(f"Boundary frames: {n_folded} folded, {n_unfolded} unfolded, "
          f"{len(state) - n_folded - n_unfolded} transition")

    print("Computing visit-based first-passage committor labels …")
    label = visit_based_committor(state)

    binned = bin_committor_2d(rg, q, label, n_bins=args.n_bins)

    frames_df = pd.DataFrame({"Rg": rg, "Q": q, "committor_label": label})
    frames_csv = os.path.join(args.out_dir, "qrg_committor_frames.csv")
    frames_df.to_csv(frames_csv, index=False)
    print(f"Wrote per-frame Q/Rg/committor data: {frames_csv}")

    box_csv = os.path.join(args.out_dir, "qrg_committor_boxes.csv")
    binned["box_committor_frames"].to_csv(box_csv, index=False)
    print(f"Wrote per-box committor data: {box_csv}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_qrg_heatmap(binned, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figure6_qrg_heatmap.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_committor_histogram(binned, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "figure7_committor_histogram.png"), dpi=150)
    plt.close(fig)
    print("Saved figure6_qrg_heatmap.png and figure7_committor_histogram.png")

    # --- Step 4: train a model to guess the odds ---------------------------
    train_df = frames_df.dropna(subset=["committor_label"])
    if len(train_df) >= 50:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        X = train_df[["Rg", "Q"]]
        y = train_df["committor_label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        model = GradientBoostingRegressor(random_state=1)
        print("Training GradientBoostingRegressor on (Rg, Q) -> committor …")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"R² (test): {score:.4f}")

        import joblib
        joblib.dump(model, os.path.join(args.out_dir, "qrg_committor_model.joblib"))
    else:
        print("Too few resolved frames to train a model; skipping step 4.")

    print("Done.")


if __name__ == "__main__":
    main()
