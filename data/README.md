# Data Directory

Large trajectory and cluster CSV files are **not stored in this repository** (excluded by `.gitignore`).

## Available cluster files

We experimented with many different cluster counts to test the sensitivity of the committor
estimates. All files below are stored on the HPC at `/scratch/mmm9886/Chignolin_Trajectory/`.

### Chignolin

| File | Clusters | Has committor (`_prob`)? |
|---|---|---|
| `cluster_40.csv` / `cluster_40_prob.csv` | 40 | ✅ |
| `cluster_112.csv` | 112 | ❌ |
| `cluster_127_prob.csv` | 127 | ✅ |
| `cluster_361.csv` / `cluster_361_prob.csv` | 361 | ✅ |
| `cluster_1120.csv` / `cluster_1120_prob.csv` | 1120 | ✅ |
| `cluster_1770.csv` / `cluster_1770_prob.csv` | 1770 | ✅ |
| `cluster_4801.csv` / `cluster_4801_prob.csv` | 4801 | ✅ |

Additional cluster files from larger trajectory experiments are in `Partial_Chignolin/`:

| File | Clusters | Has committor? |
|---|---|---|
| `cluster_400.csv` | 400 | ❌ |
| `clusters_with_folds_1528.csv` / `clusters_with_folds_1528_prob.csv` | 1528 | ✅ |
| `cluster_1339.csv` / `clusters_with_folds_1339.csv` | 1339 | ❌ |
| `cluster_2733.csv` | 2733 | ❌ |
| `cluster_3582.csv` / `cluster_3582_prob.csv` | 3582 | ✅ |

K-Means experiments (in `Kmean/`):

| File | Clusters | Has committor? |
|---|---|---|
| `kmeans-20.csv` / `kmeans-20_prob.csv` | 20 | ✅ |
| `kmeans-500.csv` / `kmeans-500_prob.csv` | 500 | ✅ |
| `kmeans-1000.csv` / `kmeans-1000_prob.csv` | 1000 | ✅ |

### WW Domain (in `WW/`)

| File | Method / cutoff | Has committor? |
|---|---|---|
| `cluster_ww_0.7.csv` / `clusters_with_folds_ww_0.7_prob.csv` | contact cutoff 0.7 | ✅ |
| `cluster_ww_0.75.csv` / `clusters_with_folds_ww_0.75_prob.csv` | contact cutoff 0.75 — **canonical** | ✅ |
| `cluster_ww_cont_100.csv` / `cluster_ww_cont_100_prob.csv` | contact map, 100 clusters | ✅ |
| `cluster_ww_cont_231.csv` / `cluster_ww_cont_231_prob.csv` | contact map, 231 clusters | ✅ |
| `clusters_with_folds_1528_ww_0.7.csv` | 1528 clusters, cutoff 0.7 | ❌ |

## Directory structure expected at runtime

```
data/
├── chignolin/
│   ├── cluster_40.csv          ← copy from HPC (see table above)
│   ├── cluster_40_prob.csv     ← output of notebooks/02_committor_counting.ipynb
│   ├── distances_ON.csv        ← output of notebooks/03_feature_extraction.ipynb
│   └── backbone_torsions.csv   ← output of notebooks/03_feature_extraction.ipynb
└── ww_domain/
    ├── clusters_with_folds_ww_0.75.csv   ← copy from HPC
    ├── clusters_with_folds_ww_0.75_prob.csv
    ├── distances_ON.csv
    └── backbone_torsions.csv
```

## Trajectory data (NYU HPC)

All trajectory files live on the NYU Abu Dhabi HPC cluster filesystem.  
They are **not publicly distributable** but can be accessed by lab members at:

### Chignolin (CLN025)

| File | Path on HPC |
|---|---|
| Trajectory (500k frames) | `/scratch/ccbg/B_DNA-Folding/Chignolin_Miles/Chignolin_Traj/f_traj/CLN025-0-protein-all_fiti_alpC.xtc` |
| Alternative all-frames file | `/scratch/ccbg/B_DNA-Folding/Chignolin_Miles/Chignolin_Traj/f_traj/all_frames_chignoling.xtc` |
| Reference topology (PDB) | `/scratch/ccbg/B_DNA-Folding/Chignolin_Miles/Chignolin_Traj/f_traj/folded_struc.pdb` |
| Topology (frame 1, for mdtraj) | `/scratch/ccbg/B_DNA-Folding/Chignolin_Miles/Chignolin_Traj/f_traj/all_frames_1_chignoling.pdb` |

### WW Domain

| File | Path on HPC |
|---|---|
| Trajectory | `/scratch/ccbg/B_DNA-Folding/Chignolin_Miles/WW_domain_Traj/all_frames_ww_domain_pbc_alpC.xtc` |
| Reference topology (PDB) | `/scratch/ccbg/B_DNA-Folding/Chignolin_Miles/WW_domain_Traj/new_selected_traj/folded_21000_NEW_ww.pdb` |

## Cluster CSV format

All cluster CSV files have the following format:

```
Time_ps,Cluster_ID
0,3
200,3
400,12
...
```

Cluster probability CSV files add a `Committor_prob` column:

```
Time_ps,Cluster_ID,Committor_prob
0,3,0.623
200,3,0.623
400,12,Null
...
```

`Null` indicates boundary frames (folded or unfolded states) or clusters  
with insufficient statistics to compute a reliable committor.

## Reproducing from scratch

Run the notebooks in order on the HPC cluster:

```bash
cd notebooks/
jupyter nbconvert --to notebook --execute --inplace 01_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_committor_counting.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_feature_extraction.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_visualization.ipynb
```

Or submit the end-to-end script via SLURM:

```bash
sbatch scripts/run_regression.sbatch
```
