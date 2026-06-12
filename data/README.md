# Data Directory

Large trajectory and cluster CSV files are **not stored in this repository** (excluded by `.gitignore`).

## Directory structure expected at runtime

```
data/
├── chignolin/
│   ├── cluster_40.csv          ← output of notebooks/01_clustering.ipynb
│   ├── cluster_40_prob.csv     ← output of notebooks/02_committor_counting.ipynb
│   ├── distances_ON.csv        ← output of notebooks/03_feature_extraction.ipynb
│   └── backbone_torsions.csv   ← output of notebooks/03_feature_extraction.ipynb
└── ww_domain/
    ├── clusters_with_folds_ww_0.75.csv
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
