"""Compare the final model with and without class weighting (experiment 'class-weighting').

REPORT next step: STAR recall (0.77) is the model's weakest point. class_weight='balanced' makes
each class count equally during training instead of letting GALAXY (59% of rows) dominate. The
question: does STAR recall rise without a larger loss elsewhere (macro F1, other recalls)?
Cross-validation on the training split only; the test set is not used.

Run from the project root:  python -m scripts.compare_class_weighting
"""
import pandas as pd

from src.config import CLASSES, FILTERS, FINAL_FEATURES, FOREST_PARAMS
from src.data import read_training_data
from src.model import build_model, cv_scores, split_data
from src.tracking import log_cv_run, start_experiment

CANDIDATES = {
    'no weighting': FOREST_PARAMS,
    'balanced': {**FOREST_PARAMS, 'class_weight': 'balanced'},
}


def main():
    data_version, df = read_training_data()
    X_train, _, y_train, _ = split_data(df, FILTERS)

    start_experiment('class-weighting')
    rows = []
    for name, params in CANDIDATES.items():
        scores = cv_scores(build_model(**params), X_train, y_train)
        log_cv_run(
            name,
            {'features': ','.join(FINAL_FEATURES), 'class_weight': params.get('class_weight', 'none'),
             **FOREST_PARAMS},
            scores,
            data_version,
        )
        rows.append({
            'candidate': name,
            'cv_macro_f1': scores['macro_f1'].mean(),
            'cv_std': scores['macro_f1'].std(),
            **{f'{c} recall': scores[f'recall_{c}'].mean() for c in CLASSES},
        })
        print(f'done: {name}')

    print()
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
