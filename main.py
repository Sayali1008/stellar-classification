"""Train and evaluate the final star/galaxy/quasar classifier (random forest on colors + r)."""
from src.config import FINAL_FEATURES, OUTPUT_DIR
from src.data import load_data
from src.evaluate import feature_importances, plot_confusion_matrix, print_report
from src.features import add_colors
from src.model import build_model, split_data, train


def main():
    df = add_colors(load_data())
    print(f'Loaded {len(df)} rows')

    X_train, X_test, y_train, y_test = split_data(df, FINAL_FEATURES)

    model = train(build_model(), X_train, y_train)
    preds = model.predict(X_test)

    print('===== final model: colors + r =====')
    print_report(y_test, preds)

    print('\nFeature importances')
    print(feature_importances(model, FINAL_FEATURES).round(3).to_string())

    OUTPUT_DIR.mkdir(exist_ok=True)
    cm_path = OUTPUT_DIR / 'confusion_matrix.png'
    plot_confusion_matrix(y_test, preds, cm_path)
    print(f'\nConfusion matrix saved to {cm_path}')


if __name__ == '__main__':
    main()
