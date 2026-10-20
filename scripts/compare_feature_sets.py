"""Compare feature sets with cross-validation and record each one in MLflow (experiment 'feature-sets').

The notebook's seven sets (analysis.ipynb, REPORT section 6) plus colors + g and colors + i from
the REPORT's next steps. Every set uses the same forest (FOREST_PARAMS) and the same 5 folds of the
training split; the test set is not used. The notebook used 200 unrestricted trees, so the numbers
differ slightly from the REPORT.

Run from the project root:  python -m scripts.compare_feature_sets
"""
import pandas as pd

from src.config import COLORS, FILTERS, FOREST_PARAMS
from src.data import read_training_data
from src.model import build_model, cv_scores, split_data
from src.tracking import log_cv_run, start_experiment

POSITION = ['alpha', 'delta']

FEATURE_SETS = {
    'colors + r': COLORS + ['r'],
    'raw filters only': FILTERS,
    'raw filters + alpha/delta': FILTERS + POSITION,
    'colors only': COLORS,
    'colors + alpha/delta': COLORS + POSITION,
    'filters + colors': FILTERS + COLORS,
    'filters + colors + alpha/delta': FILTERS + COLORS + POSITION,
    'colors + g': COLORS + ['g'],
    'colors + i': COLORS + ['i'],
}


def main():
    data_version, df = read_training_data()
    # The pipeline adds the colors itself, so it gets the raw filters (+ sky position for the
    # sets that use it). Same rows in the split as train.py: same labels, same random_state.
    X_train, _, y_train, _ = split_data(df, FILTERS + POSITION)

    start_experiment('feature-sets')
    rows = []
    for name, features in FEATURE_SETS.items():
        scores = cv_scores(build_model(features), X_train, y_train)
        log_cv_run(
            name,
            {'features': ','.join(features), 'n_features': len(features), **FOREST_PARAMS},
            scores,
            data_version,
        )
        rows.append({
            'feature set': name,
            'n': len(features),
            'cv_macro_f1': scores['macro_f1'].mean(),
            'cv_std': scores['macro_f1'].std(),
            'STAR recall': scores['recall_STAR'].mean(),
        })
        print(f'done: {name}')

    results = pd.DataFrame(rows).sort_values('cv_macro_f1', ascending=False)
    print()
    print(results.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
