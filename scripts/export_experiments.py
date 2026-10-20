"""Write EXPERIMENTS.md: one Markdown table per MLflow experiment, readable on GitHub.

Reads mlflow.db. For each run name only the latest finished run is shown, so re-running a
comparison replaces its rows instead of duplicating them (older runs stay in mlflow.db).
EXPERIMENTS.md is generated: re-run this script after new experiments instead of editing it.

Run from the project root:  python -m scripts.export_experiments
"""
from datetime import UTC, datetime

import mlflow

from src.config import CLASSES, MLFLOW_TRACKING_URI, PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / 'EXPERIMENTS.md'

# Experiment → (heading, one-line description), in the order they appear in the file
EXPERIMENTS = {
    'training': ('Training runs', 'Each `train.py` run: the model the API can serve, scored once on the held-out test set.'),
    'feature-sets': ('Feature sets', 'Which inputs to use. 5-fold CV on the training split, same forest for every set.'),
    'forest-sizes': ('Forest sizes', 'Smaller forest = smaller artifact and faster cold start. 5-fold CV on the training split.'),
    'class-weighting': ('Class weighting', "Does `class_weight='balanced'` raise STAR recall? 5-fold CV on the training split."),
}


def latest_runs(experiment):
    """Finished runs of one experiment, newest first, one per run name."""
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="attributes.status = 'FINISHED'",
        order_by=['attributes.start_time DESC'],
    )
    if runs.empty:
        return runs
    return runs.drop_duplicates('tags.mlflow.runName')


def markdown_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join(lines)


def training_table(runs):
    headers = ['Model version', 'Data version', 'Features', 'Trees', 'Min leaf', 'Test macro F1']
    headers += [f'{c} recall' for c in CLASSES]
    rows = [
        [
            run['tags.mlflow.runName'],
            f"`{run['tags.data_version'][:8]}`",
            run['params.features'],
            run['params.n_estimators'],
            run['params.min_samples_leaf'],
            f"{run['metrics.test_macro_f1']:.4f}",
            *[f"{run[f'metrics.test_recall_{c}']:.3f}" for c in CLASSES],
        ]
        for _, run in runs.iterrows()
    ]
    return markdown_table(headers, rows)


def cv_table(runs, extra_columns=()):
    """Best CV macro F1 first. extra_columns: (header, metric name, format) for single-number metrics."""
    runs = runs.sort_values('metrics.cv_macro_f1_mean', ascending=False)
    headers = ['Candidate', 'CV macro F1 (mean ± std)']
    headers += [f'{c} recall' for c in CLASSES]
    headers += [header for header, _, _ in extra_columns]
    rows = [
        [
            run['tags.mlflow.runName'],
            f"{run['metrics.cv_macro_f1_mean']:.4f} ± {run['metrics.cv_macro_f1_std']:.4f}",
            *[f"{run[f'metrics.cv_recall_{c}_mean']:.3f}" for c in CLASSES],
            *[format(run[f'metrics.{metric}'], fmt) for _, metric, fmt in extra_columns],
        ]
        for _, run in runs.iterrows()
    ]
    return markdown_table(headers, rows)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    sections = [
        '# Experiments',
        '',
        f'Generated from `mlflow.db` by `scripts/export_experiments.py` on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. '
        'Do not edit by hand. To browse every run: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.',
    ]
    for experiment, (heading, description) in EXPERIMENTS.items():
        runs = latest_runs(experiment)
        sections += ['', f'## {heading}', '', description, '']
        if runs.empty:
            sections.append('_No finished runs yet._')
        elif experiment == 'training':
            sections.append(training_table(runs))
        elif experiment == 'forest-sizes':
            sections.append(cv_table(runs, [('Size (MB)', 'size_mb', '.1f'), ('Load (s)', 'load_s', '.2f')]))
        else:
            sections.append(cv_table(runs))

    OUTPUT_PATH.write_text('\n'.join(sections) + '\n')
    print(f'Wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
