"""Train, evaluate and save the final star/galaxy/quasar classifier (random forest on colors + r).

Writes artifacts/model.joblib and artifacts/metadata.json for the API to load. Two modes:

- `python train.py`: on your machine. Records the run in MLflow (experiment 'training').
- `python train.py --deploy`: called by deploy/deploy.sh. No MLflow (deploy runs don't write
  mlflow.db). Runs the deploy gate (src/gate.py) against the live model; exits with an error if the
  new model fails it, so the script stops before building an image. A passing model is stored in S3
  under models/.
"""
import argparse
import sys

import mlflow

from src.artifacts import build_metadata, save_artifacts
from src.config import FILTERS, METADATA_PATH, MODEL_PATH, OUTPUT_DIR
from src.data import read_training_data
from src.evaluate import feature_importances, plot_confusion_matrix, print_report
from src.gate import run_gate, upload_model
from src.model import build_model, split_data, train
from src.tracking import log_training_run, start_experiment
from src.validate import validate_training_data


def train_and_save():
    """Read, validate, train, score on the test split and save. Returns what the modes need next."""
    data_version, df = read_training_data()
    print(f'Loaded {len(df)} rows')
    validate_training_data(df)

    X_train, X_test, y_train, y_test = split_data(df, FILTERS)

    model = train(build_model(), X_train, y_train)
    preds = model.predict(X_test)

    print('===== final model: colors + r =====')
    print_report(y_test, preds)

    print('\nFeature importances')
    print(feature_importances(model).round(3).to_string())

    OUTPUT_DIR.mkdir(exist_ok=True)
    cm_path = OUTPUT_DIR / 'confusion_matrix.png'
    plot_confusion_matrix(y_test, preds, cm_path)
    print(f'\nConfusion matrix saved to {cm_path}')

    metadata = build_metadata(model, y_test, preds, data_version)
    size_mb = save_artifacts(model, metadata)
    print(f'Model saved to {MODEL_PATH} ({size_mb:.1f} MB)')
    print(f"Metadata saved to {METADATA_PATH} (version {metadata['model_version']})")

    return model, metadata, X_test, y_test, cm_path


def main_local():
    start_experiment('training')

    # Everything runs inside the MLflow run: if a step fails (e.g. bad data), the run is
    # recorded as FAILED instead of disappearing
    with mlflow.start_run():
        model, metadata, _, _, cm_path = train_and_save()
        log_training_run(model, metadata, cm_path)
        print(f"MLflow run recorded: experiment 'training', run {metadata['model_version']}")


def main_deploy():
    _, metadata, X_test, y_test, _ = train_and_save()

    print()
    gate_result = run_gate(metadata, X_test, y_test)
    if not gate_result['passed']:
        sys.exit(1)
    upload_model(gate_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--deploy', action='store_true', help='deploy mode: no MLflow, run the gate')
    if parser.parse_args().deploy:
        main_deploy()
    else:
        main_local()
