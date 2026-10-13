"""Train, evaluate and save the final star/galaxy/quasar classifier (random forest on colors + r).

Writes artifacts/model.joblib and artifacts/metadata.json for the API to load.
"""
from src.artifacts import build_metadata, save_artifacts
from src.config import FILTERS, METADATA_PATH, MODEL_PATH, OUTPUT_DIR
from src.data import load_data
from src.evaluate import feature_importances, plot_confusion_matrix, print_report
from src.model import build_model, split_data, train


def main():
    df = load_data()
    print(f'Loaded {len(df)} rows')

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

    metadata = build_metadata(model, y_test, preds)
    size_mb = save_artifacts(model, metadata)
    print(f'Model saved to {MODEL_PATH} ({size_mb:.1f} MB)')
    print(f"Metadata saved to {METADATA_PATH} (version {metadata['model_version']})")


if __name__ == '__main__':
    main()
