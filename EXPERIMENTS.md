# Experiments

Generated from `mlflow.db` by `scripts/export_experiments.py` on 2026-10-20 03:20 UTC. Do not edit by hand. To browse every run: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Training runs

Each `train.py` run: the model the API can serve, scored once on the held-out test set.

| Model version | Data version | Features | Trees | Min leaf | Test macro F1 | GALAXY recall | STAR recall | QSO recall |
|---|---|---|---|---|---|---|---|---|
| 20261020-031622 | `S6MZ8CBq` | u_g,g_r,r_i,i_z,r | 50 | 5 | 0.8511 | 0.953 | 0.771 | 0.801 |

## Feature sets

Which inputs to use. 5-fold CV on the training split, same forest for every set.

| Candidate | CV macro F1 (mean ± std) | GALAXY recall | STAR recall | QSO recall |
|---|---|---|---|---|
| filters + colors + alpha/delta | 0.8572 ± 0.0019 | 0.954 | 0.781 | 0.813 |
| colors + g | 0.8546 ± 0.0015 | 0.952 | 0.777 | 0.809 |
| colors + r | 0.8531 ± 0.0029 | 0.952 | 0.776 | 0.805 |
| filters + colors | 0.8529 ± 0.0020 | 0.951 | 0.777 | 0.809 |
| colors + i | 0.8528 ± 0.0021 | 0.953 | 0.776 | 0.805 |
| colors + alpha/delta | 0.8391 ± 0.0026 | 0.946 | 0.750 | 0.791 |
| raw filters only | 0.8357 ± 0.0032 | 0.942 | 0.745 | 0.798 |
| colors only | 0.8319 ± 0.0021 | 0.940 | 0.742 | 0.787 |
| raw filters + alpha/delta | 0.8304 ± 0.0022 | 0.944 | 0.721 | 0.801 |

## Forest sizes

Smaller forest = smaller artifact and faster cold start. 5-fold CV on the training split.

| Candidate | CV macro F1 (mean ± std) | GALAXY recall | STAR recall | QSO recall | Size (MB) | Load (s) |
|---|---|---|---|---|---|---|
| n_estimators=100, min_samples_leaf=5 | 0.8545 ± 0.0021 | 0.953 | 0.778 | 0.807 | 15.0 | 0.19 |
| n_estimators=200, min_samples_leaf=5 | 0.8544 ± 0.0019 | 0.953 | 0.777 | 0.807 | 30.1 | 0.36 |
| n_estimators=200 | 0.8544 ± 0.0030 | 0.952 | 0.781 | 0.805 | 47.7 | 0.62 |
| n_estimators=50, min_samples_leaf=5 | 0.8531 ± 0.0029 | 0.952 | 0.776 | 0.805 | 7.5 | 0.09 |

## Class weighting

Does `class_weight='balanced'` raise STAR recall? 5-fold CV on the training split.

| Candidate | CV macro F1 (mean ± std) | GALAXY recall | STAR recall | QSO recall |
|---|---|---|---|---|
| no weighting | 0.8531 ± 0.0029 | 0.952 | 0.776 | 0.805 |
| balanced | 0.8496 ± 0.0022 | 0.917 | 0.810 | 0.842 |
