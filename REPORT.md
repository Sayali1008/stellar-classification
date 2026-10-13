# Classifying Stars, Galaxies and Quasars from Imaging Data

**Dataset:** Sloan Digital Sky Survey, Data Release 17 (SDSS17), 100,000 spectroscopically classified objects
**Final model:** Random forest on four colors plus `r`-band brightness
**Result:** Test macro F1 of **0.852**, compared with **0.25** for an always-GALAXY baseline

---

## 1. Summary

The goal is to classify objects as stars, galaxies or quasars using **only imaging data**: the brightness of each object through five color filters. A model like this could label newly photographed objects that haven't been observed spectroscopically, which applies to most of the sky.

A random forest using four colors (`u−g`, `g−r`, `r−i`, `i−z`) and one brightness measurement (`r`) reached a macro F1 of 0.852 on a held-out test set, close to its cross-validation estimate of 0.854 ± 0.003. Stars and galaxies were the most-confused pair. The strongest single feature was `r−i`, which separates galaxies from the other two classes.

## 2. Data

- **Source:** SDSS17, 100,000 rows × 18 columns.
- **Unit of observation:** one row is one spectrum of one object. `spec_obj_ID` is the unique key.
- **Target (`class`):**

| Class | Share |
|---|---|
| GALAXY | 59.45% |
| STAR | 21.59% |
| QSO (quasar) | 18.96% |

**Cleaning**
- **ID columns** (`obj_ID`, `spec_obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `plate`, `fiber_ID`) were loaded as strings. They label things and aren't measurements.
- **`obj_ID` isn't a reliable identifier.** When stored as a float, rounding merged different objects into the same value.
- **One row was dropped** because it had the placeholder value −9999 in `u`, `g` and `z`, meaning no measurement.
- **No missing values.** Sky coordinates are within valid ranges.

## 3. Problem framing

**Inputs:** only data available at the imaging stage.
- The five filter magnitudes: `u, g, r, i, z`.
- Four colors derived from them: `u_g, g_r, r_i, i_z`. A color is the difference between two adjacent filters. It describes the shape of an object's light and doesn't depend on distance.

**Excluded**
- **`redshift`** comes from the spectrum, so it isn't available for a newly photographed object. Using it would be leakage.
- **ID and metadata columns** don't describe the object physically.
- **`alpha`/`delta` (sky position)** were tested as candidates (Section 6) and rejected.

**Metric:** macro F1. The model is meant for a full catalog, so every class matters equally. Accuracy would reward a model that mostly predicts GALAXY.

**Setup:** a stratified 80/20 train/test split (`random_state=42`). The baseline always predicts GALAXY, which gives 59.45% accuracy.

## 4. Exploratory findings

Per-class medians (standard deviations in parentheses):

| Class | `u_g` | `g_r` | `r_i` | `i_z` | `r` |
|---|---|---|---|---|---|
| GALAXY | 1.64 (1.23) | 1.40 (0.63) | **0.72** (0.40) | 0.39 (0.35) | 20.10 (1.87) |
| STAR | 1.32 (0.94) | 0.53 (0.66) | 0.21 (0.61) | 0.12 (0.55) | **18.96** (1.97) |
| QSO | **0.39** (0.84) | 0.21 (0.45) | 0.15 (0.39) | 0.16 (0.40) | 20.77 (1.08) |

- **QSOs have low `u_g`:** quasars emit strongly in ultraviolet (UV excess).
- **Galaxies have high `r_i`:** they are red in `r − i`.
- **STAR sits between the other two classes in most colors and has a wide spread.** Stars range from hot and blue to cool and red, so their colors form a long, curved band rather than a compact cluster.
- **Stars are brighter** (lower `r` magnitude) because they're nearby, inside our own galaxy.

**Hypothesis H1:** in the `u_g` vs `g_r` plane, stars and galaxies overlap the most, so STAR ↔ GALAXY should be the most-confused pair.

## 5. Model comparison

All three models used all nine features (5 filters + 4 colors). Logistic regression was scaled inside a pipeline.

| Model | Macro F1 | GALAXY recall | STAR recall | QSO recall |
|---|---|---|---|---|
| Baseline (always GALAXY) | 0.25 | 1.00 | 0.00 | 0.00 |
| Logistic regression | 0.67 | 0.90 | 0.32 | 0.80 |
| Random forest | **0.85** | 0.95 | 0.78 | 0.80 |

**Why the forest wins:**
- **Logistic regression** gives each class one region with straight edges. Stars form a curved band in color space, with some stars on each side of the galaxies, and a straight-edged region can't follow that shape. As a result, logistic regression found only 32% of stars, and most of the stars it missed were labeled GALAXY.
- **The random forest** builds its boundaries from many small splits, so it can follow curved and irregular shapes. It found 78% of stars.
- **QSO recall is the same (0.80) for both models.** UV excess can be separated with a single straight cut, so the extra flexibility adds nothing for that class.

## 6. Feature selection

Seven feature sets were compared with **5-fold stratified cross-validation** on the training data only, using the same random forest settings for each.

| Feature set | Features | Mean macro F1 | Std |
|---|---|---|---|
| filters + colors + alpha/delta | 11 | 0.859 | 0.002 |
| **colors + r** | **5** | **0.854** | **0.003** |
| filters + colors | 9 | 0.854 | 0.002 |
| colors + alpha/delta | 6 | 0.842 | 0.003 |
| raw filters + alpha/delta | 7 | 0.840 | 0.002 |
| raw filters only | 5 | 0.839 | 0.002 |
| colors only | 4 | 0.829 | 0.004 |

**Findings**
- **Colors + r ties filters + colors with 5 features instead of 9.** The four colors plus any one filter can be used to recalculate all five filters, so the extra brightness columns add no new information.
- **Adding brightness helps:** colors only scores 0.829 and colors + r scores 0.854. Brightness separates stars from galaxies where their colors overlap.
- **Colors beat raw filters (0.854 vs 0.839) even though they carry the same information.** A tree splits on one feature at a time. A rule like "`u − g` < 1" is a diagonal boundary in raw-filter space, which trees can only approximate with many small steps. Providing the color directly turns it into one clean split.
- **Sky position (alpha/delta) was rejected.** Adding it raised macro F1 by about 0.005, consistently in all five folds. However, the gain varies between feature sets (+0.013 on colors only, +0.001 on raw filters), which looks more like a pattern in SDSS sky coverage and targeting than a physical signal. Random cross-validation folds share the same parts of the sky, so they can't rule this out, and a gain like this may not carry over to new regions.

**Choice:** random forest on **colors + r**: the simplest set in the top tier, using only physical measurements.

## 7. Final results

The final model was trained on all training rows and scored once on the test set.

**Test macro F1: 0.852** (cross-validation estimate 0.854 ± 0.003, within one standard deviation)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| GALAXY | 0.92 | 0.95 | 0.93 | 11,889 |
| STAR | 0.85 | 0.77 | 0.81 | 4,319 |
| QSO | 0.82 | 0.80 | 0.81 | 3,792 |
| **Macro avg** | 0.86 | 0.84 | **0.85** | 20,000 |

**Confusion matrix** (rows = actual, columns = predicted; row percentages in parentheses):

| Actual ↓ / Predicted → | GALAXY | STAR | QSO |
|---|---|---|---|
| **GALAXY** | 11,321 (95.2%) | 275 (2.3%) | 293 (2.5%) |
| **STAR** | 606 (14.0%) | 3,338 (77.3%) | 375 (8.7%) |
| **QSO** | 440 (11.6%) | 301 (7.9%) | 3,051 (80.5%) |

**H1 is confirmed.** STAR ↔ GALAXY is the most-confused pair (881 errors in both directions), followed by GALAXY ↔ QSO (733) and STAR ↔ QSO (676). Most errors for both STAR and QSO go to GALAXY. That's the majority class pulling uncertain cases: where the classes overlap, galaxies outnumber the others about 3 to 1 in the training data.

**Feature importances**

| Feature | Importance |
|---|---|
| `r_i` | 0.307 |
| `g_r` | 0.210 |
| `u_g` | 0.183 |
| `i_z` | 0.156 |
| `r` | 0.144 |

- **`r_i` ranks first, not `u_g`.** It separates galaxies from everything else. Physically, galaxies have a sharp drop in their light at 4000 Å. At this sample's typical galaxy redshift (median about 0.45), that drop falls within the `r` filter, which makes galaxies red in `r − i`.
- **Importance depends on class size.** A feature that sorts galaxies, 59% of the data, gets more credit than `u_g`, which mainly sorts quasars (19%). A lower importance doesn't mean a feature is less useful for its own class.
- **`r` ranks last, but it still matters.** Adding it raised macro F1 by 0.025, mostly by fixing STAR vs GALAXY cases where the colors overlap.

## 8. Limitations

- **The class mix isn't the real sky.** It reflects which objects SDSS chose to take spectra of, and quasars are over-represented. The model's preference for GALAXY on uncertain cases comes from that mix.
- **The model may partly learn how SDSS selected targets.** SDSS chose quasar targets partly by UV color and used different brightness cutoffs for each class. Some of the predictive power of `u_g` and `r` may therefore come from those selection rules rather than from physics.
- **Sky coverage is mostly northern** (`delta` from −19° to +83°). Performance on other regions is untested.
- **A few objects were observed more than once** (one confirmed pair). That could put the same object in both the training and test sets, though the effect is negligible at this scale.
- **The test set was viewed during exploration.** Feature sets were also scored on the test set before the final choice. The choice relied on cross-validation, but the reported test score may be slightly optimistic.
- **No class balancing.** The imbalance is moderate and per-class recall was acceptable, so balancing wasn't applied. STAR recall (0.77) is the weakest point.

## 9. Next steps

- **Spatial cross-validation:** group the folds by sky region to test directly whether the alpha/delta gain comes from survey coverage.
- **Permutation importance on the test set,** to confirm the importance ranking without the training-data bias.
- **Class weighting** (`class_weight='balanced'`), to see whether STAR recall can rise without a larger drop in precision.
- **Colors + other single filters** (`g`, `i`), to check whether `r` is the best brightness measurement or just a reasonable default.

## 10. Reproduce

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the SDSS17 stellar classification CSV into `data/star_classification.csv`.
3. Run the pipeline:
   ```bash
   python train.py
   ```

It prints the classification report, macro F1 and feature importances, saves the confusion matrix to `outputs/confusion_matrix.png`, and saves the model and its metadata to `artifacts/`.
