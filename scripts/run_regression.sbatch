#!/bin/bash
#SBATCH -J committor_regression
#SBATCH -o logs/regression_%j.out
#SBATCH -e logs/regression_%j.err
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Activate conda environment
CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail

# ── configurable ──────────────────────────────────────────────────────────
SYSTEM="chignolin"       # or "ww_domain"
N_CLUSTERS=40            # cluster count used in committor CSV
DOWN=1                   # downsampling fraction (1 = no downsampling)
# ─────────────────────────────────────────────────────────────────────────

mkdir -p logs

python - <<EOF
import yaml, sys, pandas as pd

sys.path.insert(0, ".")

with open(f"config/${SYSTEM}.yaml") as f:
    cfg = yaml.safe_load(f)

from src.ONH import ONH
from src.generate_pair import pair
from src.extract_features import features
from src.regression import regression
from src.plot_importances import plot_importances, plot_shap, plot_true_vs_pred

import os
os.makedirs("results", exist_ok=True)

PDB    = cfg["topology_pdb"]
FILES  = cfg["trajectory_xtc"]
STRIDE = cfg["stride"]
SYSTEM = cfg["system"]

print("Extracting features …")
A, B = ONH(PDB)
atom_pairs = pair(A, B)
df_dist = features(PDB, FILES, pairs=atom_pairs, al="custom_distances", stride=STRIDE)
df_tors = features(PDB, FILES, al="backbone_torsions", stride=STRIDE)
df_x    = pd.concat([df_dist, df_tors], axis=1)

committor_csv = f"data/{SYSTEM}/cluster_${N_CLUSTERS}_prob.csv"
score, model, X_train, X_test, y_train, y_test = regression(df_x, committor_csv, down=${DOWN})

plot_importances(model, filename=f"results/feature_importance.jpg")
plot_shap(model, X_test, filename=f"results/shap_summary.jpg")
plot_true_vs_pred(y_train, model.predict(X_train), filename="results/true_vs_pred_train.jpg", title="Train")
plot_true_vs_pred(y_test,  model.predict(X_test),  filename="results/true_vs_pred_test.jpg",  title="Test")

print(f"Done. Test R² = {score:.4f}")
EOF
