"""Point MLflow at the project's mlflow.db, choose the experiment a run belongs to, and record runs."""
import mlflow

from src.config import (
    FOREST_PARAMS,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

# metadata.json's metric names → shorter MLflow metric names
METRIC_NAMES = {'precision': 'precision', 'recall': 'recall', 'f1-score': 'f1'}


def start_experiment(name):
    """Make `name` the active experiment, creating it on first use.

    The artifact location can only be set when an experiment is created, so a new experiment is created here with its files in S3.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=f'{MLFLOW_ARTIFACT_ROOT}/{name}')
    mlflow.set_experiment(name)


def log_training_run(model, metadata, confusion_matrix_path):
    """Record one train.py run: settings, data version, test metrics, plot, metadata and the model.

    Call inside `with mlflow.start_run():`. Metrics come from metadata, so MLflow and
    metadata.json always show the same numbers.
    """
    forest = model.named_steps['model']

    # Run name = model version, so a run in the UI matches the model the API reports in /health
    mlflow.set_tags({
        'mlflow.runName': metadata['model_version'],
        'model_version': metadata['model_version'],
        'data_version': metadata['data_version'],
    })

    # Read the parameters from the trained model to know whatt were used for this model's training
    forest_params = forest.get_params()
    mlflow.log_params({
        'features': ','.join(forest.feature_names_in_),
        **{name: forest_params[name] for name in FOREST_PARAMS},
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
    })

    # Held-out test metrics only: training-set scores would look great and mean nothing
    test_metrics = metadata['test_metrics']
    mlflow.log_metric('test_macro_f1', test_metrics['macro_f1'])
    for cls, scores in test_metrics['per_class'].items():
        mlflow.log_metrics({f'test_{short}_{cls}': scores[name] for name, short in METRIC_NAMES.items()})
    
    mlflow.log_artifact(str(confusion_matrix_path))
    mlflow.log_dict(metadata, 'metadata.json')
    # The exact file the Lambda image serves, so the run records what was deployed
    mlflow.log_artifact(str(MODEL_PATH))


def log_cv_run(name, params, scores, data_version, extra_metrics=None):
    """Record one cross-validated candidate as its own run in the active experiment.

    `scores` comes from model.cv_scores: one array of fold scores per metric. Each is logged as
    cv_<metric>_mean and cv_<metric>_std. `extra_metrics` adds single numbers (e.g. file size).
    """
    with mlflow.start_run(run_name=name):
        mlflow.set_tag('data_version', data_version)
        mlflow.log_params(params)
        for metric, folds in scores.items():
            mlflow.log_metrics({f'cv_{metric}_mean': folds.mean(), f'cv_{metric}_std': folds.std()})
        if extra_metrics:
            mlflow.log_metrics(extra_metrics)
