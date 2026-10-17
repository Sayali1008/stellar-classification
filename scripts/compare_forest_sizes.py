"""Compare random forest sizes: cross-validated macro F1 vs. saved file size and load time.

A smaller forest means a smaller artifact and a faster Lambda cold start. The test set is not
used here; the chosen setting goes into FOREST_PARAMS in src/config.py.

Run from the project root:  python -m scripts.compare_forest_sizes
"""
import io
import time

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import FILTERS, RANDOM_STATE
from src.data import read_training_data
from src.model import build_model, split_data

CANDIDATES = [
    {'n_estimators': 200},
    {'n_estimators': 200, 'min_samples_leaf': 5},
    {'n_estimators': 100, 'min_samples_leaf': 5},
    {'n_estimators': 50, 'min_samples_leaf': 5},
]


def artifact_size_and_load_time(model):
    """Size (MB) of the model saved the way train.py saves it, and how long it takes to load."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    size_mb = buffer.tell() / 1e6
    buffer.seek(0)
    start = time.perf_counter()
    joblib.load(buffer)
    return size_mb, time.perf_counter() - start


def main():
    data_version, df = read_training_data()
    X_train, _, y_train, _ = split_data(df, FILTERS)

    # Same 5 folds for every candidate; stratified so each fold keeps the class mix
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for params in CANDIDATES:
        scores = cross_val_score(build_model(**params), X_train, y_train, cv=cv, scoring='f1_macro')
        size_mb, load_s = artifact_size_and_load_time(build_model(**params).fit(X_train, y_train))
        rows.append({
            'params': str(params),
            'cv_macro_f1': scores.mean(),
            'cv_std': scores.std(),
            'size_mb': size_mb,
            'load_s': load_s,
        })
        print(f'done: {params}')

    results = pd.DataFrame(rows)
    print()
    print(results.round(4).to_string(index=False))

    # Smallest artifact whose CV score is within one standard deviation of the best
    best = results.loc[results['cv_macro_f1'].idxmax()]
    good_enough = results[results['cv_macro_f1'] >= best['cv_macro_f1'] - best['cv_std']]
    choice = good_enough.loc[good_enough['size_mb'].idxmin()]
    print(f"\nBest CV macro F1: {best['cv_macro_f1']:.4f} ± {best['cv_std']:.4f} ({best['params']})")
    print(f"Smallest within one std: {choice['params']} "
          f"(CV {choice['cv_macro_f1']:.4f}, {choice['size_mb']:.1f} MB)")


if __name__ == '__main__':
    main()
