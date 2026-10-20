"""Train, evaluate and save the final star/galaxy/quasar classifier (random forest on colors + r).

Writes artifacts/model.joblib and artifacts/metadata.json for the API to load, and records the run
in MLflow (experiment 'training').
"""
import mlflow

from src.artifacts import build_metadata, save_artifacts
from src.config import FILTERS, METADATA_PATH, MODEL_PATH, OUTPUT_DIR
from src.data import read_training_data
from src.evaluate import feature_importances, plot_confusion_matrix, print_report
from src.model import build_model, split_data, train
from src.tracking import log_training_run, start_experiment
from src.validate import validate_training_data


def main():
    start_experiment('training')

    # Everything runs inside the MLflow run: if a step fails (e.g. bad data), the run is
    # recorded as FAILED instead of disappearing
    with mlflow.start_run():
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

        log_training_run(model, metadata, cm_path)
        print(f"MLflow run recorded: experiment 'training', run {metadata['model_version']}")


if __name__ == '__main__':
    main()
