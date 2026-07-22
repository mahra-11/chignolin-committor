# Protein Folding Committor Prediction

Predict the **committor probability** of each conformation in molecular dynamics (MD) trajectories of two proteins — **Chignolin** (CLN025, 10 residues) and the **WW domain** — using machine learning.

The committor q(i) is the probability that the system, if released from conformation i, will reach the **folded state** before the **unfolded state**. It is the gold-standard measure of progress along a folding reaction coordinate. We predict it from structural features (pairwise O/N distances and backbone torsions) using a LightGBM regressor, then use SHAP values to reveal which atomic distances best characterize the transition state.

---

## Method Overview

```
MD Trajectory (.xtc)
        │
        ▼
 01_clustering.ipynb
   TICA dimensionality reduction
   K-Means clustering → cluster_N.csv
        │
        ▼
 02_committor_counting.ipynb
   "Count-once" first-passage counting
   → cluster_N_prob.csv  (Committor_prob column)
        │
        ▼
 03_feature_extraction.ipynb
   O/N atom identification (ONH.py)
   Pairwise distance + backbone torsion features
   → distances_ON.csv, backbone_torsions.csv
        │
        ▼
 04_regression.ipynb
   LightGBM regression (+ stratified downsampling)
   SHAP feature importance
   → R² score, importance plots, SHAP summary
        │
        ▼
 05_visualization.ipynb
   2D free energy surfaces
   Committor-colored contour plots
```

---

## Repository Structure

```
chignolin-committor/
├── src/                         Python source modules
│   ├── ONH.py                   Identify O/N/H atoms in a PDB
│   ├── generate_pair.py         Generate O/N atom-index pairs
│   ├── extract_features.py      PyEMMA trajectory featurization
│   ├── committor.py             Committor counting + CSV writing
│   ├── downsample.py            Stratified downsampling by committor bin
│   ├── regression.py            LightGBM regression pipeline
│   ├── plot_importances.py      SHAP + importance visualization
│   └── contours.py              Free energy + committor contour plots
├── notebooks/
│   ├── 01_clustering.ipynb      TICA + K-Means clustering
│   ├── 02_committor_counting.ipynb  Committor probability estimation
│   ├── 03_feature_extraction.ipynb  O/N distance + torsion features
│   ├── 04_regression.ipynb      LightGBM training + SHAP analysis
│   └── 05_visualization.ipynb   Free energy and committor plots
├── config/
│   ├── chignolin.yaml           Data paths + state IDs for Chignolin
│   └── ww_domain.yaml           Data paths + state IDs for WW domain
├── structures/
│   └── chignolin/               Local structure files for visualization
│       ├── Chignolin_Aligned.pdb
│       ├── conf_CLN025.gro
│       └── conf.gro
├── data/
│   └── README.md                ← Data access guide (files not in repo)
├── scripts/
│   └── run_regression.sbatch    End-to-end SLURM job (NYU HPC)
├── requirements.txt
└── README.md
```

---

## Q/Rg Committor (no clustering)

`src/qrg_committor.py` + `scripts/run_qrg_committor.py` implement the
alternative committor estimate described in the project presentation
("Changing to Q"). Clustering turned out to (a) lump geometrically distinct
conformers into the same cluster and (b) get dominated by the folded state,
requiring an RMSD filter that discarded >60% of frames. This method instead:

1. Computes two continuous order parameters per frame — **Q** (fraction of
   native contacts, Best et al. 2013 smoothed formula) and **Rg** (radius of
   gyration) — with no clustering step at all.
2. Bins frames onto a 2D (Rg, Q) grid.
3. Labels each frame with a **visit-based** first-passage rule: walking
   forward along the trajectory, does it hit the folded boundary
   (Q ≥ 0.9 by default) or the unfolded boundary (Q ≤ 0.1) first?
4. Averages the per-frame labels within each grid box to get a box-level
   committor estimate, then trains a regressor (`GradientBoostingRegressor`)
   to predict it from (Rg, Q).

```bash
python scripts/run_qrg_committor.py --config config/chignolin.yaml
```

Outputs land in `results/`: `figure6_qrg_heatmap.png` (frame-count heat map,
reproducing Figure 6), `figure7_committor_histogram.png` (committor-value
histogram weighted by frame count, reproducing Figure 7), the per-frame and
per-box CSVs, and the trained model.

---

## Data

Trajectory data lives on the **NYU NYUAD HPC cluster** and is not stored in this repository.  
See [`data/README.md`](data/README.md) for file paths and instructions on reproducing the cluster CSVs.

**Chignolin trajectory:**  
`/scratch/ccbg/B_DNA-Folding/Chignolin_Miles/Chignolin_Traj/f_traj/CLN025-0-protein-all_fiti_alpC.xtc`  
Reference PDB: `.../f_traj/folded_struc.pdb`

**WW domain trajectory:**  
`/scratch/ccbg/B_DNA-Folding/Chignolin_Miles/WW_domain_Traj/all_frames_ww_domain_pbc_alpC.xtc`  
Reference PDB: `.../new_selected_traj/folded_21000_NEW_ww.pdb`

---

## Setup

```bash
# On NYU HPC: load conda
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh

# Create environment
conda create -y -n committor_env python=3.9
conda activate committor_env
pip install -r requirements.txt
```

---

## Quick Start

### Interactive (Jupyter)

```bash
conda activate committor_env
jupyter lab notebooks/
```

Run notebooks 01 → 05 in order.  
Edit the `yaml` config path in each notebook to switch between Chignolin and WW domain.

### Batch (SLURM)

```bash
# Edit SYSTEM, N_CLUSTERS, and DOWN at the top of the script if needed
sbatch scripts/run_regression.sbatch
tail -f logs/regression_<JOBID>.out
```

---

## Key Results

### Chignolin (40 clusters, O/N distances + backbone torsions)

| Split | R² |
|---|---|
| Train | ~0.93 |
| Test  | ~0.85 |

Top SHAP features are hydrogen-bond distances across the β-hairpin turn, especially:
- `THR 8 O – TYR 2 N` (backbone H-bond at turn apex)
- `TYR 2 N – TRP 9 N` (cross-strand contact)
- `ASP 3 O – THR 6 N` (β-hairpin H-bond)

### WW Domain (1120 clusters, contact cutoff 0.75)

Top features involve contacts between the three β-strands, analogous to the Chignolin pattern.

---

## Module Reference

| Module | Key function | Description |
|---|---|---|
| `src/ONH.py` | `ONH(pdb)` | Returns O/N atom indices and polar-H indices |
| `src/generate_pair.py` | `pair(A, B)` | All unique (0-based) atom-index pairs |
| `src/extract_features.py` | `features(pdb, files, pairs, al, stride)` | PyEMMA featurization → DataFrame |
| `src/committor.py` | `compute_committor(csv, folded, unfolded)` | Count-once committor estimation |
| `src/committor.py` | `write_committor_csv(...)` | Annotate cluster CSV with Committor_prob |
| `src/downsample.py` | `dsample(df, down)` | Stratified downsampling by committor bin |
| `src/regression.py` | `regression(df_x, committor_csv, down)` | LightGBM training → (R², model, splits) |
| `src/plot_importances.py` | `plot_importances / plot_shap / plot_true_vs_pred` | Save analysis figures |
| `src/contours.py` | `plot_contours / plot_free_energy_2d` | Free energy + committor visualizations |

